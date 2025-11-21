import pandas as pd
import numpy as np
import random
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.features.features import Sequence

class HuggingFaceIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(
        self,
        name : str,
        datasets : dict,
        datasetPath : str,
        datasetConfig : dict,
        seed : int = 42,
        maxNaN : float = 0.25,
        minUniqueElementRatio : float = 0.02,
    ):
        random.seed(seed)
        self.__seed : int = seed
        self.__datasets : dict = datasets
        self.__datasetConfig : dict = datasetConfig
        self.__name : str = name
        self.__sampleSize : int = 1
        self.__datasetPath : str = datasetPath
        self.__maxNaN : float = maxNaN
        self.__minUniqueElementRatio : float = minUniqueElementRatio

        self.__features : dict = {}
        self.__datasetSizes : dict = {}

        self.__datasetsLoader : dict = {}
        self.__splittedDatasets : dict = {subDataset : {} for subDataset in self.__datasets}
        self.__indexIterator : dict = {subDataset : {} for subDataset in self.__datasets}

        self.__buffer : dict = {subDataset : [] for subDataset in self.__datasets}

    def __preproces(self, sequence : list) -> list:
        """
        Method to preprocess NaN and zeros
        """
        mean : float = np.mean(sequence)
        if np.isnan(sequence).sum() / len(sequence) > self.__maxNaN:
            return []

        if (
            len(
                {element : None for element in sequence}
            ) / len(sequence)
        ) < self.__minUniqueElementRatio:
            return []

        if all(element - mean < 1e-2 and element - mean > -1e-2 for element in sequence): # Discard straight lines
            return []

        if all(element == 0.0 for element in sequence):
            return []

        sequenceNumpy : np.ndarray = np.array(sequence)
        nans : np.ndarray = np.isnan(sequenceNumpy)
        not_nans : np.ndarray = ~nans
        indices : np.ndarray = np.arange(len(sequenceNumpy))

        sequenceNumpy[nans] = np.interp(indices[nans], indices[not_nans], sequenceNumpy[not_nans])

        return sequenceNumpy.tolist()

    def setSampleSize(self, sampleSize : int):
        """
        Method to set sample size
        """
        self.__sampleSize = sampleSize

    def getAvailableFeatures(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
        if subdataset in self.__features:
            return self.__features[subdataset]

        if subdataset not in self.__datasetsLoader:
            self.__datasetsLoader[subdataset] = load_dataset(self.__datasetConfig["datasetName"], subdataset, split="train", cache_dir=self.__datasetPath)

        features : dict = {}

        index : int = 1
        features["index"] = 0
        for feature in list(self.__datasetsLoader[subdataset].features.keys()):
            if isinstance(self.__datasetsLoader[subdataset].features[feature], Sequence):
                features[feature] = index

            index += 1

        return features

    def getDatasetMetadata(self) -> dict:
        return {"std" : 1.0, "mean" : 0.0}

    def getSeasonabilityError(self, subdataset : str) -> float:
        """
        Method to get seasonability error
        """
        return 1.0

    def resetIteration(self, subdataset : str, randomOrder : bool = False, trainPartition : float = 1.0):
        """
        Method to reset dataset iteration
        """
        if subdataset not in self.__datasetsLoader:
            self.__datasetsLoader[subdataset] = load_dataset(self.__datasetConfig["datasetName"], subdataset, split="train", cache_dir=self.__datasetPath)

        splittedDataset :  DatasetDict = self.__datasetsLoader[subdataset].train_test_split(test_size=1-trainPartition, shuffle=randomOrder, seed=self.__seed)

        self.__splittedDatasets[subdataset] = {
            "train" : splittedDataset["train"],
            "test" : splittedDataset["test"],
        }

        self.__indexIterator[subdataset] = {
            "train" : [index for index in range(len(splittedDataset["train"]))],
            "test" : [index for index in range(len(splittedDataset["test"]))],
        }

    def releaseIterator(self, subdataset : str):
        """
        Method to release iterator
        """
        if subdataset in self.__datasetsLoader:
            del self.__datasetsLoader[subdataset]

        if subdataset in self.__splittedDatasets:
            del self.__splittedDatasets[subdataset]

        if subdataset in self.__indexIterator:
            del self.__indexIterator[subdataset]

        if subdataset in self.__buffer:
            del self.__buffer[subdataset]

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
            return None

        sample : pd.core.frame.Dataframe = self.loadSample(
            subdataset=subdataset,
            sampleIndex=self.__indexIterator[subdataset][category][-1],
            features=features,
            category=category,
        )

        if len(self.__buffer[subdataset]) == 0:
            self.__indexIterator[subdataset][category].pop()

        return sample

    def loadSample(
            self,
            subdataset : str,
            sampleIndex : int,
            sampleSize : int = 1,
            features : list = [],
            category : str = "train",
        ) -> pd.core.frame.DataFrame:
        """
        Method to load sample
        """
        if subdataset not in self.__datasetsLoader:
            self.__datasetsLoader[subdataset] = load_dataset(
                self.__datasetConfig["datasetName"],
                subdataset,
                split="train",
                cache_dir=self.__datasetPath,
            )

        if len(features) == 0:
            features = [key for key in self.getAvailableFeatures(subdataset)]

        if len(self.__indexIterator[subdataset][category]) == 0 or self.__sampleSize >= len(self.__indexIterator[subdataset][category]):
            return None

        featuresIndices : list = [index for index in range(len(features))]

        if len(self.__buffer[subdataset]) == 0:
            sampleDict : dict = self.__datasetsLoader[subdataset][sampleIndex]
            bufferElementTemplate : dict = {feature : [] for feature in featuresIndices}
            bufferElementTemplate["index"] = list(range(0, self.__sampleSize))
            for featureIndex in range(1, len(featuresIndices)):
                feature : int = featuresIndices[featureIndex]
                index : int = 0

                if type(sampleDict[features[feature]][0]) == list:
                    for sequence in sampleDict[features[feature]]:
                        for indexSample in range(0, len(sequence), self.__sampleSize):
                            if len(self.__buffer[subdataset]) <= index:
                                self.__buffer[subdataset].append(bufferElementTemplate.copy())

                            preprocessedSequence : list = self.__preproces(
                                sequence[indexSample:indexSample+self.__sampleSize].copy()
                            )
                            if len(preprocessedSequence) == 0:
                                continue
                            self.__buffer[subdataset][index][feature] = preprocessedSequence
                            index += 1
                elif(
                    type(sampleDict[features[feature]][0]) == int or type(sampleDict[features[feature]][0]) == float
                ):
                    for indexSample in range(0, len(sampleDict[features[feature]]), self.__sampleSize):
                        if len(self.__buffer[subdataset]) <= index:
                            self.__buffer[subdataset].append(bufferElementTemplate.copy())

                        preprocessedSequence : list = self.__preproces(
                            sampleDict[features[feature]][indexSample:indexSample+self.__sampleSize].copy()
                        )
                        if len(preprocessedSequence) == 0:
                            continue
                        self.__buffer[subdataset][index][feature] = preprocessedSequence
                        index += 1

            for index in range(len(self.__buffer[subdataset])):
                for feature in self.__buffer[subdataset][index]:
                    nanSequence : list = [np.nan for _  in range(0, self.__sampleSize - len(self.__buffer[subdataset][index][feature]))]
                    self.__buffer[subdataset][index][feature] += nanSequence

        sampleBuffer : dict = self.__buffer[subdataset].pop(0)

        sample : pd.core.frame.DataFrame = pd.DataFrame(sampleBuffer)

        return sample