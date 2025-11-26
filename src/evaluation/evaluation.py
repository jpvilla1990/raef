import pandas as pd
import numpy as np
import concurrent.futures
import random
from datasetsModule.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from model.chronosModel import Chronos
from utils.fileSystem import FileSystem
from utils.utils import Utils
from datasetsModule.datasetIterator import DatasetIterator

class Evaluation(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()
        random.seed(self._getConfig()["seed"])
        self.__dataset : Datasets = Datasets()

    def __getMASE(self, seasonabilityError : float, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate MEAN ABSOLUTE SCALED ERROR
        """
        meanAbsoluteError : float = np.mean(
            abs(prediction - groundTruth),
        )

        return meanAbsoluteError / seasonabilityError
    
    def __getMAE(self, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate MEAN ABSOLUTE ERROR
        """
        meanAbsoluteError : float = np.mean(
            abs((prediction - groundTruth)),
        )

        return meanAbsoluteError

    def __getMSE(self, groundTruth : np.ndarray, prediction : np.ndarray, std : np.ndarray | None = None) -> float:
        """
        Method to calculate MEAN SQUARED ERROR
        """
        if std:
            return np.mean(
                ((prediction - groundTruth) / std) ** 2,
            )
        else:
            return np.mean(
                ((prediction - groundTruth)) ** 2,
            )

    def evaluateChronosRagLeveling(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
        bolt : bool = True,
    ) -> dict:
        """
        Method to evaluate model RAG CA
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : Chronos = Chronos(bolt=bolt, frozen=False)
        model.setInputSpaceCollection(collection, dataset)

        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        maxTestSamplesPerSubdataset : int = int(maxTestSamples / len(subdatasets))
        reportMAE : np.ndarray = np.array([])
        reportMSE : np.ndarray = np.array([])
        reportMSERef : np.ndarray = np.array([])
        reportMASE : np.ndarray = np.array([])
        reportMASEExtended : np.ndarray = np.array([])
        reportMASERef : np.ndarray = np.array([])
        reportMASERaf : np.ndarray = np.array([])
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                metadata : dict = iterator.getDatasetMetadata()
                seasonabilityError : float = iterator.getSeasonabilityError(element)
                std : float = metadata["std"]
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while running:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            trainSet,
                        )
                        sample : pd.core.frame.DataFrame = futureSample.result()
                        if sample is None:
                            break
                        if len(sample) < predictionLength + contextLength:
                            break

                        indexes : list = [index for index in range(1,len(features))]
                        random.shuffle(indexes)
                        for i in range(len(indexes)):
                            index : int = indexes[i]
                            if sample[index].isna().any().any():
                                continue
                            if (sample[index] == 0.0).any().any():
                                continue
                            if (sample[index] == sample[index].mean().mean()).all().all():
                                continue

                            pred : np.ndarray = None
                            refPred : np.ndarray = None
                            rafPred : np.ndarray = None
                            with concurrent.futures.ThreadPoolExecutor() as executor2:
                                futurePred : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRaef,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                    extended=False,
                                )
                                futurePredExtended : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRaef,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                    extended=True,
                                )
                                futurePredRef : concurrent.futures._base.Future = executor2.submit(
                                    model.predict,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                )
                                futurePredRaf : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRaf,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                )
                                pred = futurePred.result()
                                predExtended = futurePredExtended.result()
                                refPred = futurePredRef.result()
                                rafPred = futurePredRaf.result()

                            Utils.plot(
                                [
                                    sample[index].tolist(),
                                    sample[index].iloc[:contextLength].to_list() + pred.tolist(),
                                ],
                                "ground_truth_pred.png",
                                "-",
                                contextLength,
                            )
                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            maseExtended : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                predExtended,
                            )

                            maseRef : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                refPred,
                            )

                            maseRaf : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                rafPred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                                std,
                            )

                            mseRef : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                refPred,
                                std,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if maseExtended:
                                reportMASEExtended = np.append(reportMASEExtended, [maseExtended])
                            if maseRef:
                                reportMASERef = np.append(reportMASERef, [maseRef])
                            if maseRaf:
                                reportMASERaf = np.append(reportMASERaf, [maseRaf])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])
                            if mseRef:
                                reportMSERef = np.append(reportMSERef, [mseRef])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return {
            "base": np.mean(reportMASERef),
            "raf": np.mean(reportMASERaf),
            "incremental": np.mean(reportMASE),
            "raef": np.mean(reportMASEExtended),
        }

    def evaluateMoiraiMoERagCA(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
        inputSpace : bool = True,
        useTrainRagDatabase : bool = False,
        fineTunedModel : str = "",
    ) -> dict:
        """
        Method to evaluate model RAG CA
        """
        ragDataset : str = "lotsaData" if useTrainRagDatabase else dataset
        collection = f"moiraiMoETrainingRafL2_{contextLength}_{predictionLength}" if useTrainRagDatabase else collection
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        modelFineTuned : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
            loadFineTunedModel=True,
            fineTunedModel=fineTunedModel,
        )
        modelRaf : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        if inputSpace:
            model.setInputSpaceCollection(collection, ragDataset)
            modelFineTuned.setInputSpaceCollection(collection, ragDataset)
            modelRaf.setInputSpaceCollection(collection, ragDataset)
        else:
            model.setEmbeddingSpaceCollection(collection, ragDataset)
            modelFineTuned.setEmbeddingSpaceCollection(collection, ragDataset)
            modelRaf.setEmbeddingSpaceCollection(collection, ragDataset)
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        maxTestSamplesPerSubdataset : int = int(maxTestSamples / len(subdatasets))
        reportMAE : np.ndarray = np.array([])
        reportMSE : np.ndarray = np.array([])
        reportMSERefBase : np.ndarray = np.array([])
        reportMSERaf : np.ndarray = np.array([])
        reportMASE : np.ndarray = np.array([])
        reportMASEExtended : np.ndarray = np.array([])
        reportMASERefBase : np.ndarray = np.array([])
        reportMASERaf : np.ndarray = np.array([])
        reportMASEFineTuning : np.ndarray = np.array([])
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                metadata : dict = iterator.getDatasetMetadata()
                seasonabilityError : float = iterator.getSeasonabilityError(element)
                std : float = metadata["std"]
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while running:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            trainSet,
                        )
                        sample : pd.core.frame.DataFrame = futureSample.result()
                        if sample is None:
                            break
                        if len(sample) < predictionLength + contextLength:
                            break

                        indexes : list = [index for index in range(1,len(features))]
                        random.shuffle(indexes)
                        for i in range(len(indexes)):
                            index : int = indexes[i]
                            if sample[index].isna().any().any():
                                continue
                            if (sample[index] == 0.0).any().any():
                                continue
                            if (sample[index] == sample[index].mean().mean()).all().all():
                                continue

                            pred : np.ndarray = None
                            refPred : np.ndarray = None
                            with concurrent.futures.ThreadPoolExecutor() as executor2:
                                futurePred : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRaef,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                    extended=False,
                                )
                                futurePredExtended : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRaef,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                    extended=True,
                                )
                                futurePredRefBase : concurrent.futures._base.Future = executor2.submit(
                                    model.inference,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                )
                                futurePredRaf : concurrent.futures._base.Future = executor2.submit(
                                    modelRaf.rafInference,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                    False,
                                    False,
                                )
                                futurePredFineTuning : concurrent.futures._base.Future = executor2.submit(
                                    modelFineTuned.inference,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                )
                                pred = futurePred.result()
                                predExtended = futurePredExtended.result()
                                refPredBase = futurePredRefBase.result()
                                futurePredRaf = futurePredRaf.result()
                                futurePredFineTuning = futurePredFineTuning.result()

                            Utils.plot(
                                [
                                    sample[index].tolist(),
                                    sample[index].iloc[:contextLength].to_list() + pred.tolist(),
                                ],
                                "ground_truth_pred.png",
                                "-",
                                contextLength,
                            )
                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            maseExtended : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                predExtended,
                            )

                            maseRefBase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                refPredBase,
                            )

                            maseRaf : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                futurePredRaf,
                            )

                            maseFineTuning : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                futurePredFineTuning,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                                std,
                            )

                            mseRefBase : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                refPredBase,
                                std,
                            )

                            mseRaf : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                futurePredRaf,
                                std,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if maseExtended:
                                reportMASEExtended = np.append(reportMASEExtended, [maseExtended])
                            if maseRefBase:
                                reportMASERefBase = np.append(reportMASERefBase, [maseRefBase])
                            if maseRaf:
                                reportMASERaf = np.append(reportMASERaf, [maseRaf])
                            if maseFineTuning:
                                reportMASEFineTuning = np.append(reportMASEFineTuning, [maseFineTuning])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])
                            if mseRefBase:
                                reportMSERefBase = np.append(reportMSERefBase, [mseRefBase])
                            if mseRaf:
                                reportMSERaf = np.append(reportMSERaf, [mseRaf])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return {
            "base": np.mean(reportMASERefBase),
            "fineTuning": np.mean(reportMASEFineTuning),
            "raf": np.mean(reportMASERaf),
            "incremental": np.mean(reportMASE),
            "raef": np.mean(reportMASEExtended),
        }