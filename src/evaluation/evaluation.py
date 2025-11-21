import pandas as pd
import numpy as np
import concurrent.futures
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from datasetsModule.datasets import Datasets
from model.lineal import LinealRegression
from model.moiraiMoe import MoiraiMoE
from model.chronosModel import Chronos
from model.chatTime import ChatTimeModel
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

    def __loadReport(self, report : str) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()[report]
        )
        return reports if type(reports) == dict else dict()

    def __writeReport(self, entry : dict, report : str):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()[report],
            self.__loadReport(report) | entry,
        )

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

    def compileReports(self, reportOriginName : str = "evaluationReportsMoiraiMoE", reportTargetName : str = "evaluationFinalReport"):
        """
        Method to compile results in a human readable report
        """
        report : dict = self.__loadReport(reportOriginName)

        tables : dict = {}

        for dataset in report:
            for scenario in report[dataset]:
                if scenario not in tables:
                    tables.update({
                        scenario : {
                            "scenario" : {},
                            "indices" : [],
                        },
                    })
                if dataset not in tables[scenario]["indices"]:
                    tables[scenario]["indices"].append(dataset)

                totalIterations : int = 0
                for subdataset in report[dataset][scenario]:
                    for metric in report[dataset][scenario][subdataset]:
                        if metric == "numberIterations":
                            continue
                        if metric not in tables[scenario]["scenario"]:
                            tables[scenario]["scenario"].update({
                                metric : [],
                            })
                        if len(tables[scenario]["scenario"][metric]) < len(tables[scenario]["indices"]):
                            tables[scenario]["scenario"][metric].append(
                                report[dataset][scenario][subdataset][metric]["mean"],
                            )
                        else:
                            tables[scenario]["scenario"][metric][-1] = ((tables[scenario]["scenario"][metric][-1] * totalIterations) + (report[dataset][scenario][subdataset][metric]["mean"] * report[dataset][scenario][subdataset]["numberIterations"])) / (totalIterations + report[dataset][scenario][subdataset]["numberIterations"])
                    totalIterations += report[dataset][scenario][subdataset]["numberIterations"]

                if "numberIterations" not in tables[scenario]["scenario"]:
                    tables[scenario]["scenario"].update({
                        "numberIterations" : [],
                    })

                tables[scenario]["scenario"]["numberIterations"].append(totalIterations)

        elements : list = []
        doc : SimpleDocTemplate = SimpleDocTemplate(self._getFiles()[reportTargetName], pagesize=letter)
        for scenario in tables:
            df : pd.core.frame.DataFrame = pd.DataFrame(tables[scenario]["scenario"], index=tables[scenario]["indices"]).round(6)
            elements.append(Table([[f"{reportOriginName} Context Lenght, Prediction Lenght = {scenario}"]], colWidths=[400]))

            tableData = [["Index"] + df.columns.tolist()]
            for index, row in df.iterrows():
                tableData.append([index] + row.tolist())

            table : Table = Table(tableData)

            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])
            table.setStyle(style)

            elements.append(table)
            elements.append(Table([[""]], colWidths=[400]))  # Add space between tables

        doc.build(elements)

    def evaluateLinearModel(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : LinealRegression = LinealRegression(
            predictionLength = predictionLength,
            contextLength = contextLength,
        )
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.predictWithStepFitting(
                                sample[index].iloc[:contextLength].values,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                            )
                            if pred is None:
                                continue

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsLineal")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                print(f"MASE: {reportMASE.mean()}")

                self.__writeReport(report, "evaluationReportsLineal")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoE(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.inference(sample[[0, index]].iloc[:contextLength], dataset)
                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoE")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoE")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoERagMean(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        model.setRagCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.ragInference(sample[[0, index]].iloc[:contextLength], dataset)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERagMean")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoERagMean")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoERagSoftMax(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        cosine : bool = True,
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        model.setRagCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.ragInference(sample[[0, index]].iloc[:contextLength], dataset, True, cosine)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERagSoftMax")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoERagSoftMax")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoERafSoftMax(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        model.setRafCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.rafInference(sample[[0, index]].iloc[:contextLength], dataset, True)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERafSoftMax")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoERafSoftMax")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoERafCosSoftMax(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        model.setRafCosCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.rafInference(sample[[0, index]].iloc[:contextLength], dataset, True, True)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERafCosSoftMax")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoERafCosSoftMax")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateRagPrediction(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        model.setRagCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.ragOnlyInference(sample[[0, index]].iloc[:contextLength], dataset, True)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERagOnlySoftMax")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsMoiraiMoERagOnlySoftMax")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateChatTimes(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : ChatTimeModel = ChatTimeModel(
            predictionLength = predictionLength,
            contextLength = contextLength,
        )
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.inference(sample[[index]].iloc[:contextLength])

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsChatTime")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsChatTime")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateChatTimesRag(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : ChatTimeModel = ChatTimeModel(
            predictionLength = predictionLength,
            contextLength = contextLength,
            collectionName = collection,
        )
        model.setRagCollection(collection, dataset)
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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())
                seasonabilityError : float = iterator.getSeasonabilityError(element)

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

                            pred : np.ndarray = model.ragInference(sample[[index]].iloc[:contextLength], dataset)

                            mase : float = self.__getMASE(
                                seasonabilityError,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsChatTimeRag")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
                    "MASE" : {
                        "mean" : float(reportMASE.mean()),
                        "median" : float(np.median(reportMASE)),
                    },
                    "MAE" : {
                        "mean" : float(reportMAE.mean()),
                        "median" : float(np.median(reportMAE)),
                    },
                    "MSE" : {
                        "mean" : float(reportMSE.mean()),
                        "median" : float(np.median(reportMSE)),
                    },
                    "numberIterations" : iterations,
                }

                self.__writeReport(report, "evaluationReportsChatTimeRag")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateChronosRagLeveling(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
        raf : bool = True,
        loadPretrainedModel : bool = False,
        bolt : bool = True,
    ) -> dict:
        """
        Method to evaluate model RAG CA
        """
        report : dict = {}
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}, collection : {collection}_{dataset}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : Chronos = Chronos(bolt=bolt, frozen=False, loadPretrainedModel=loadPretrainedModel)
        model.setRafCollection(collection, dataset)

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
                                    model.predictRag,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                    extended=False,
                                )
                                futurePredExtended : concurrent.futures._base.Future = executor2.submit(
                                    model.predictRag,
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
                raise e
                print("Exception: " + str(e))
                continue

        print("MASE: " + str(np.mean(reportMASE)))
        print("MASE Extended: " + str(np.mean(reportMASEExtended)))
        print("MASE Ref: " + str(np.mean(reportMASERef)))
        print("MASE Raf: " + str(np.mean(reportMASERaf))) 

    def evaluateChronos(
        self,
        contextLength : int,
        predictionLength : int,
        dataset : str,
        subdataset : str = "",
        trainSet : bool = False,
    ) -> dict:
        """
        Method to evaluate model RAG CA
        """
        report : dict = {}
        print(f"Evaluating Dataset {dataset}, context length : {contextLength}, prediction length : {predictionLength}")
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        model : Chronos = Chronos(model="amazon/chronos-t5-small")

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
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
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
                            with concurrent.futures.ThreadPoolExecutor() as executor2:
                                futurePred : concurrent.futures._base.Future = executor2.submit(
                                    model.predict,
                                    sample[index].iloc[:contextLength].values,
                                    predictionLength,
                                )
                                pred = futurePred.result()

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

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                                std,
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

                print("MASE: " + str(np.mean(reportMASE)))

            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        return report

    def evaluateMoiraiMoERagCA(
        self,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
        dataset : str,
        collection : str,
        subdataset : str = "",
        trainSet : bool = False,
        raf : bool = True,
        useTrainRagDatabase : bool = False,
        loadPretrainedRagCA : bool = True,
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
            loadPretrainedModel=loadPretrainedRagCA,
        )
        modelFineTuned : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
            loadPretrainedModel=False,
            loadFineTunedModel=True,
            fineTunedModel=fineTunedModel,
        )
        if raf:
            model.setRafCollection(collection, ragDataset)
            modelFineTuned.setRafCollection(collection, ragDataset)
        else:
            model.setRagCollection(collection, ragDataset)
            modelFineTuned.setRagCollection(collection, ragDataset)
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        modelRaf : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        #if "cosine" in collection:
        #    modelRaf.setRafCollection(collection.replace("Cosine","RafL2"), "lotsaData")
        #elif "L2" in collection and "RafL2" not in collection:
        #    modelRaf.setRafCollection(collection.replace("L2","RafL2"), "lotsaData")
        #else:
        #    modelRaf.setRafCollection(collection, "lotsaData")
        if raf:
            modelRaf.setRafCollection(collection, ragDataset)
        else:
            modelRaf.setRafCollection(f"moiraiMoETrainingRafL2_{contextLength}_{predictionLength}", ragDataset)

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
                                    model.ragCaInference,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                    cosine=False,
                                    extended=False,
                                )
                                futurePredExtended : concurrent.futures._base.Future = executor2.submit(
                                    model.ragCaInference,
                                    sample[[0, index]].iloc[:contextLength],
                                    dataset,
                                    cosine=False,
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
                raise e
                print("Exception: " + str(e))
                continue
        print("MASE: " + str(np.mean(reportMASE)))
        print("MASE Extended: " + str(np.mean(reportMASEExtended)))
        print("MASE Ref Base: " + str(np.mean(reportMASERefBase)))
        print("MASE Raf: " + str(np.mean(reportMASERaf)))
        print("MASE Fine Tuning: " + str(np.mean(reportMASEFineTuning)))