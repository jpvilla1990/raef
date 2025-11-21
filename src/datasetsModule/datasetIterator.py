import random
import math
import pandas as pd
import numpy as np
from exceptions.datasetException import DatasetException
from utils.utils import Utils

class DatasetIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict, datasetConfig : dict, seed : int = 42):
        random.seed(seed)
        self.__datasets : dict = datasets
        self.__datasetConfig : dict = datasetConfig
        self.__name : str = name
        self.__sampleSize : int = 1

        self.__features : dict = {}
        self.__datasetSizes : dict = {}

        self.__indexIterator : dict = {subDataset : {} for subDataset in self.__datasets}

    def __str__(self) -> str:
        return self.__name

    def __getDatasetMetada(self, subdataset : str) -> dict:
        """
        Method to get available features
        """

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        dataframe : pd.core.frame.DataFrame = Utils.loadPandasFromArrow(self.__datasets[subdataset])

        features : list = dataframe.columns.tolist()

        mean = dataframe[features[1:]].stack().mean()
        sdt = dataframe[features[1:]].stack().std()

        return {
            "numberFeatures" : len(features),
            "numberObservations" : len(dataframe),
            "mean" : mean,
            "std" : sdt,
        }

    def getSeasonabilityError(self, subdataset : str) -> float:
        """
        Method to get seasonability error
        """
        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        seasonality : int = self.__datasetConfig["seasonality"][subdataset]

        dataframe : pd.core.frame.DataFrame = Utils.loadPandasFromArrow(self.__datasets[subdataset])
        features : list = dataframe.columns.tolist()[1:]
        seasonal_errors : list = []

        for col in features:
            series : np.ndarray = dataframe[col].dropna().to_numpy()

            if len(series) <= seasonality:
                continue  # Not enough data to compute seasonality

            # Compute |y_t - y_{t - m}|
            diffs : np.ndarray = np.abs(series[seasonality:] - series[:-seasonality])
            seasonal_errors.append(np.mean(diffs))

        # Return mean seasonal error across all series
        return float(np.mean(seasonal_errors)) if seasonal_errors else float('nan')

    def setSampleSize(self, sampleSize : int):
        """
        Method to set sample size
        """
        self.__sampleSize = sampleSize

    def getDatasetMetadata(self) -> dict:
        """
        Method to get sizes of all datasets
        """
        self.__datasetMetadata = {subDataset : self.__getDatasetMetada(subDataset) for subDataset in self.__datasets}
        self.__datasetMetadata["mean"] = 0.0
        self.__datasetMetadata["std"] = 0.0

        totalSamples : int = 0
        for subDataset in self.__datasets:
            samples : int = self.__datasetMetadata[subDataset]["numberFeatures"] * self.__datasetMetadata[subDataset]["numberObservations"]
            self.__datasetMetadata["mean"] += self.__datasetMetadata[subDataset]["mean"] * samples
            self.__datasetMetadata["std"] += self.__datasetMetadata[subDataset]["std"] * samples
            totalSamples += samples

        self.__datasetMetadata["mean"] /= totalSamples
        self.__datasetMetadata["std"] /= totalSamples

        return self.__datasetMetadata

    def getAvailableFeatures(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
        features : dict = {}

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        features = {value : index for index, value in enumerate(pd.read_csv(
            self.__datasets[subdataset],
            nrows=0,
            sep=self.__datasetConfig["separator"],
            decimal=self.__datasetConfig["decimal"],
        ).columns)}

        return features

    def loadSample(
            self,
            subdataset : str,
            sampleIndex : int = 0,
            sampleSize : int = 1,
            features : list = [],
        ) -> pd.core.frame.DataFrame:
        """
        Method to load samples
        """
        self.__features[subdataset] = self.getAvailableFeatures(subdataset)

        self.__datasetSizes[subdataset] = self.__getDatasetMetada(subdataset)
        sample : pd.core.frame.DataFrame = None
        featuresIndices : list = None
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        if len(features) == 0:
            featuresIndices = [self.__features[subdataset][key] for key in self.__features[subdataset]]
        else:
            featuresIndices = [self.__features[subdataset][feature] for feature in features]

        if sampleIndex <= 0:
            sampleIndex : int = random.randint(1, maxNumberSamples - sampleSize + 1)
        elif sampleIndex > maxNumberSamples - sampleSize + 1:
            sampleSize = maxNumberSamples - sampleIndex + 1

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )
        if sampleSize < 1:
            raise DatasetException(
                f"The sampleSize can not be less than 1"
            )

        if "frame" not in self.__indexIterator[subdataset]:
            self.__indexIterator[subdataset]["frame"] = Utils.loadPandasFromArrow(self.__datasets[subdataset]).iloc[:, featuresIndices]
            self.__indexIterator[subdataset]["frame"].columns = featuresIndices

        if sampleSize >= maxNumberSamples:
            sample = self.__indexIterator[subdataset]["frame"]

        else:
            sampleIndices : list = range(sampleIndex, sampleIndex + sampleSize)
            sample = self.__indexIterator[subdataset]["frame"].iloc[sampleIndex-1:sampleIndex + sampleSize-1]
            sample.index = sampleIndices

        return sample

    def resetIteration(self, subdataset : str, randomOrder : bool = False, trainPartition : float = 1.0):
        """
        Method to reset dataset iteration
        """
        self.__datasetSizes[subdataset] = self.__getDatasetMetada(subdataset)
        self.__indexIterator[subdataset] = {}
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        indexIterator : list = [index for index in range(1, maxNumberSamples - self.__sampleSize + 2)]

        if randomOrder:
            random.shuffle(indexIterator)
        else:
            indexIterator.sort(reverse=True)

        border : int = math.floor(trainPartition * len(indexIterator))

        self.__indexIterator[subdataset] = {
            "train" : indexIterator[:border],
            "test" : indexIterator[border:],
        }

    def iterateDataset(
            self,
            subdataset : str,
            features : list = [],
            train : bool = True,
        ) -> pd.core.frame.DataFrame:
        """
        Method to iterate through out the whole dataset
        """
        category : str = "test"
        if train:
            category = "train"

        if len(self.__indexIterator[subdataset][category]) == 0:
            if "frame" in self.__indexIterator[subdataset]:
                del self.__indexIterator[subdataset]["frame"]
            return None

        sample : pd.core.frame.Dataframe = self.loadSample(
            subdataset=subdataset,
            sampleIndex=self.__indexIterator[subdataset][category][-1],
            sampleSize=self.__sampleSize,
            features=features,
        )

        self.__indexIterator[subdataset][category].pop()

        return sample
