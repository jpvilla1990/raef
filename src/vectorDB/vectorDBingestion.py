import pandas as pd
import numpy as np
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasetsModule.datasets import Datasets
from datasetsModule.datasetIterator import DatasetIterator
from model.moiraiMoe import MoiraiMoE

class VectorDBIngestion(FileSystem):
    """
    Class to handle vector DB ingestion for all datasets
    """
    def __init__(self):
        super().__init__()
        self.__dataset : Datasets = Datasets()

    def __loadDatabaseTracking(self) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()["databaseTracking"]
        )
        return reports if type(reports) == dict else dict()

    def __writeDatabaseTracking(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()["databaseTracking"],
            self.__loadDatabaseTracking() | entry,
        )

    def ingestDatasetMoiraiMoE(
        self,
        dataset : str,
        collectionName : str,
        contextLength : int,
        predictionLength : int,
        inputSpace : bool = False,
        train : bool = True,
    ):
        """
        Method to ingest dataset in a collection for moiraiMoE
        """
        print(f"Ingesting dataset {dataset} in collection {collectionName}_{dataset}")
        maxNumberSamples : int = self._getConfig()["vectorDatabase"]["maxNumberSamples"]
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = 100,
        )
        if inputSpace:
            model.setInputSpaceCollection(collectionName, dataset)
        else:
            model.setEmbeddingSpaceCollection(collectionName, dataset)
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        datasetConfig : dict = Utils.readYaml(
            self._getFiles()["datasets"]
        )
        subdatasets : list = list(datasetConfig[dataset].keys())

        iterations : int = 0

        try:
            model.deleteDataset(dataset)
        except Exception as e:
            print("Exception: " + str(e))
        maxSamplesPerSubdataset : int = int(maxNumberSamples / len(subdatasets))
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                sampleNumber : int = 0
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())

                running : bool = True
                while running:
                    sample : pd.core.frame.DataFrame = iterator.iterateDataset(element, features, train=train)
                    if sample is None:
                        break
                    if len(sample) < predictionLength + contextLength:
                        break

                    for index in range(1,len(features)):
                        if sample[index].isna().any().any():
                            continue
                        if (sample[index] == 0.0).any().any():
                            continue
                        if (sample[index] == sample[index].mean().mean()).all().all():
                            continue

                        model.ingestVector(
                            sample[index].iloc[:contextLength].values,
                            sample[index].iloc[contextLength:contextLength+predictionLength].values,
                            dataset,
                        )

                        iterations += 1
                        sampleNumber += 1

                        if sampleNumber >= maxSamplesPerSubdataset:
                            running = False
                            break

                if iterations <= 0:
                    continue

            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        databaseTracking : dict = self.__loadDatabaseTracking()
        if f"{collectionName}_{dataset}" not in databaseTracking:
            databaseTracking[f"{collectionName}_{dataset}"] = {}

        databaseTracking[f"{collectionName}_{dataset}"][dataset] = iterations
        self.__writeDatabaseTracking(databaseTracking)

    def ingestDatasetsMoiraiMoE(self, collection : str, inputSpace : bool = False, train : bool = True):
        """
        Method to ingest all datasets to MoiraiMoE
        """
        collections : dict = self._getConfig()["vectorDatabase"]["collections"][collection]

        for dataset in collections["datasets"]:
            databaseTracking : dict = self.__loadDatabaseTracking()
            collectionDataset : str = f"{collection}_{dataset}"
            if collectionDataset in databaseTracking:
                if dataset in databaseTracking[collectionDataset]: # Skip if the dataset is already ingested in collection
                    pass
                    #continue

            model : MoiraiMoE = MoiraiMoE(
                predictionLength = collections["prediction"],
                contextLength = collections["context"],
                numSamples = 100,
            )
            if inputSpace:
                model.setInputSpaceCollection(collection, dataset)
            else:
                model.setEmbeddingSpaceCollection(collection, dataset)
            model.deleteDataset(dataset)
            model.deleteCollection(collection, dataset)

            self.ingestDatasetMoiraiMoE(
                dataset,
                collection,
                collections["context"],
                collections["prediction"],
                inputSpace,
                train,
            )
