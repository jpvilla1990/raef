from typing import Callable
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from raef.utils.fileSystem import FileSystem

class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Class to handle custom embedding functions
    """
    def __init__(self, embedingFunction : Callable):
        super().__init__()
        self.__embedingFunction : Callable = embedingFunction

    def __call__(self, input : Documents) -> Embeddings:
        inputArray : np.ndarray = np.array([float(element) for element in input[0].split(",")])
        return self.__embedingFunction(inputArray).numpy()[0]

class vectorDB(FileSystem):
    """
    Class to handle vector database
    """
    def __init__(self):
        super().__init__()
        self.__chromaClient : chromadb.api.client.Client = chromadb.PersistentClient(
            path=self._getPaths()["vectorDatabase"],
            settings=Settings(anonymized_telemetry=False)
        )

    def setCollection(
            self,
            collection : str,
            dataset : str,
            embeddingFunction : Callable = None,
            similarity_metric : str = None,
        ):
        """
        Method to set collection
        """
        if similarity_metric is None:
            similarity_metric = self._getConfig()["vectorDatabase"]["collections"][collection]['metric']

        self.__collection : chromadb.api.models.Collection.Collection = self.__chromaClient.get_or_create_collection(
            name=f"{collection}_{dataset}",
            embedding_function=CustomEmbeddingFunction(embeddingFunction),
            metadata=similarity_metric
        )

    def deleteCollection(
            self,
            collectionName : str,
            dataset : str,
        ):
        """
        Method to set collection
        """
        self.__chromaClient.delete_collection(
            name=f"{collectionName}_{dataset}",
        )

    def ingestTimeseries(self, context : np.ndarray, prediction : np.ndarray, dataset : str = ""):
        """
        Method to ingest time series in collection
        """
        contextStr : str = ",".join(map(str, context.tolist()))
        predictionStr : str = ",".join(map(str, prediction.tolist()))
        id = str(uuid.uuid4())
        self.__collection.add(
            ids=[id],
            documents=[contextStr],
            metadatas=[{
                "prediction" : predictionStr,
                "dataset" : dataset,
            }]
        )

    def deleteDataset(self, dataset : str):
        """
        Method to delete dataset from collection
        """
        self.__collection.delete(
            where={"dataset" : dataset}
        )

    def updateMetadata(self, ids : [str], metadatas : [dict]):
        """
        Method to update metadata of several elements in collection
        """
        self.__collection.update(
            ids=ids,
            metadatas=metadatas,
        )

    def getAllCollection(self) -> dict:
        """
        Method to get all collections in vector database
        """
        return self.__collection.get()

    def getSample(self, id : str) -> dict:
        """
        Method to get sample from collection by id
        """
        return self.__collection.get(
            ids=[id],
        )

    def queryTimeseries(self, query : np.ndarray, k : int = 1) -> tuple:
        """
        Method to query element from vector database

        return list[context + prediction], scores
        """
        queryStr : str = ",".join(map(str, query.tolist()))
        queried : dict = {}

        queried = self.__collection.query(
            n_results=k,
            query_texts=[queryStr],
        )

        documents : list = queried["documents"][0]
        metadatas : list = queried["metadatas"][0]
        scores : list = []
        if self.__collection.metadata["hnsw:space"] == "cosine":
            scores = [1 - distance for distance in queried["distances"][0]]
        else:
            scores = queried["distances"][0]

        predictions : list = [metadata["prediction"] for metadata in metadatas]

        output : list = [f"{documents[index]},{predictions[index]}" for index in range(len(documents))]

        return (
            np.array([[float(sample) for sample in element.split(",")] for element in output]),
            scores,
        )