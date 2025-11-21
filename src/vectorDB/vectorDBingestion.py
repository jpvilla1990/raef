import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasetsModule.datasets import Datasets
from datasetsModule.datasetIterator import DatasetIterator
from model.moiraiMoe import MoiraiMoE
from model.chatTime import ChatTimeModel

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

    def assign_labels_in_batches(self, X : np.ndarray, centroids : np.ndarray, batch_size : int =10000):
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]                         # (batch_size, d)

            # squared distances = ||x||^2 + ||c||^2 - 2 x·c
            # (more memory-friendly than forming broadcasted 3D array)
            X_norm2 = np.sum(X_batch**2, axis=1)[:, None]  # (batch_size,1)
            C_norm2 = np.sum(centroids**2, axis=1)[None, :]# (1,k)
            cross    = X_batch @ centroids.T              # (batch_size,k)
            dists2   = X_norm2 + C_norm2 - 2*cross         # (batch_size,k)

            labels[start:end] = np.argmin(dists2, axis=1)

        return labels

    def resampled_kmeans(self, X, k, r, n_iter=5):
        """
        Inner loop:
        - X: data points (n_samples, n_features)
        - k: number of clusters
        - r: number of points to sample from each cluster per iteration
        - n_iter: how many refinement iterations
        """
        # 1. Initial k-means
        km = KMeans(n_clusters=k, n_init=1)
        km.fit(X)
        centroids = km.cluster_centers_
        labels = km.labels_

        for it in range(n_iter):
            # 2. Sample r points per cluster
            sampled_points = []
            for c in range(k):
                cluster_points = X[labels == c]
                if len(cluster_points) == 0:
                    continue
                # choose up to r points (or all if cluster smaller than r)
                idx = np.random.choice(len(cluster_points), 
                    size=min(r, len(cluster_points)),
                    replace=False
                )
                sampled_points.append(cluster_points[idx])
        
            sampled_points = np.vstack(sampled_points)

            # 3. Run k-means on sampled points
            km = KMeans(n_clusters=k, n_init=1)
            km.fit(sampled_points)
            centroids = km.cluster_centers_

            # 4. Reassign ALL original data to new centroids
            # (this is just computing nearest centroid)
            labels = self.assign_labels_in_batches(X, centroids, batch_size=10000)

        return centroids, labels

    def find_nearest_in_batches(self, X : np.ndarray, centroids : np.ndarray, batch_size=10000):
        k = centroids.shape[0]
        d = centroids.shape[1]
        best_dist = np.full(k, np.inf, dtype=np.float64)
        best_idx = np.full(k, -1, dtype=np.int64)
    
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            X_batch = X[start:end]     # (batch_size, d)

            # Compute distances between this batch and all centroids
            # (batch_size, k)
            dists = np.sum((X_batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

            # For each centroid, see if there’s a closer point in this batch
            for j in range(k):
                # find the closest point in the batch for this centroid
                local_idx = np.argmin(dists[:, j])
                local_dist = dists[local_idx, j]

                if local_dist < best_dist[j]:
                    best_dist[j] = local_dist
                    best_idx[j] = start + local_idx

        return best_idx

    def hierarchical_resampled_kmeans(self, X : np.ndarray ,clusters : int, max_iter=10):
        """
        Outer loop:
        - X: original dataset
        - k_list: list of k per level, e.g. [100000, 10000, 1000]
        - r_list: list of r per level (sample size per cluster)
        - n_iter_list: iterations per level
        Returns:
            centroids_per_level: list of centroids at each level
            labels_per_level:   list of label arrays mapping each point in X to clusters
        """
        centroids, labels = self.resampled_kmeans(X, clusters, 1, max_iter)

        nearest_indices = self.find_nearest_in_batches(X, centroids)

        return X[nearest_indices]

    def ingestDatasetMoiraiMoEKmeans(
        self,
        dataset : str,
        collectionName : str,
        contextLength : int,
        predictionLength : int,
        raf : bool = False,
        train : bool = True,
    ):
        print(f"Ingesting dataset {dataset} in collection {collectionName}_{dataset}")
        maxNumberSamples : int = self._getConfig()["vectorDatabase"]["maxNumberSamples"]
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = 100,
        )
        if raf:
            model.setRafCollection(collectionName, dataset)
        else:
            model.setRagCollection(collectionName, dataset)
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

        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                datasetArray : np.ndarray = np.zeros((contextLength + predictionLength,))

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

                        datasetArray = np.vstack((datasetArray, sample[index].values))

                print(datasetArray.shape)
                subdataset : np.ndarray = None
                subdatasetMaxNumberSample : int = int(maxNumberSamples / len(subdatasets))
                if len(datasetArray) <= subdatasetMaxNumberSample:
                    subdataset = datasetArray
                else:
                    subdataset : np.ndarray = self.hierarchical_resampled_kmeans(datasetArray, clusters=subdatasetMaxNumberSample)
                print(subdataset.shape)

                for index in range(subdataset.shape[0]):
                    model.ingestVector(
                        subdataset[index,:contextLength],
                        subdataset[index,contextLength:contextLength+predictionLength],
                        dataset,
                    )
                    iterations += 1

            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        databaseTracking : dict = self.__loadDatabaseTracking()
        if f"{collectionName}_{dataset}" not in databaseTracking:
            databaseTracking[f"{collectionName}_{dataset}"] = {}

        databaseTracking[f"{collectionName}_{dataset}"][dataset] = iterations
        self.__writeDatabaseTracking(databaseTracking)

    def ingestDatasetMoiraiMoE(
        self,
        dataset : str,
        collectionName : str,
        contextLength : int,
        predictionLength : int,
        raf : bool = False,
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
        if raf:
            model.setRafCollection(collectionName, dataset)
        else:
            model.setRagCollection(collectionName, dataset)
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

    def ingestDatasetsMoiraiMoE(self, collection : str, raf : bool = False, train : bool = True):
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
            if raf:
                model.setRafCollection(collection, dataset)
            else:
                model.setRagCollection(collection, dataset)
            model.deleteDataset(dataset)
            model.deleteCollection(collection, dataset)

            self.ingestDatasetMoiraiMoE(
                dataset,
                collection,
                collections["context"],
                collections["prediction"],
                raf,
                train,
            )

    def ingestDatasetChatTime(self, dataset : str, collectionName : str, contextLength : int, predictionLength : int):
        """
        Method to ingest dataset in a collection using chat time
        """
        print(f"Ingesting dataset {dataset} in collection {collectionName}_{dataset}")
        maxNumberSamples : int = self._getConfig()["vectorDatabase"]["maxNumberSamples"]
        model : ChatTimeModel = ChatTimeModel(
            predictionLength = predictionLength,
            contextLength = contextLength,
            collectionName = collectionName,
        )
        model.setRagCollection(collectionName, dataset)
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
                    sample : pd.core.frame.DataFrame = iterator.iterateDataset(element, features, train=True)
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
                print("Exception: " + str(e))
                continue

        databaseTracking : dict = self.__loadDatabaseTracking()
        if f"{collectionName}_{dataset}" not in databaseTracking:
            databaseTracking[f"{collectionName}_{dataset}"] = {}

        databaseTracking[f"{collectionName}_{dataset}"][dataset] = iterations
        self.__writeDatabaseTracking(databaseTracking)

    def ingestDatasetsChatTime(self, collection : str):
        """
        Method to ingest all datasets to MoiraiMoE
        """
        collections : dict = self._getConfig()["vectorDatabase"]["collections"][collection]

        for dataset in collections["datasets"]:
            databaseTracking : dict = self.__loadDatabaseTracking()
            collectionDataset : str = f"{collection}_{dataset}"
            if collectionDataset in databaseTracking:
                if dataset in databaseTracking[collectionDataset]: # Skip if the dataset is already ingested in collection
                    continue

            self.ingestDatasetChatTime(
                dataset,
                collection,
                collections["context"],
                collections["prediction"],
            )
