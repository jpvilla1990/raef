import pandas as pd
from utils.utils import Utils
from utils.fileSystem import FileSystem
from utils.timeManager import TimeManager

class MonashPreparer(FileSystem):
    """
    Class to Handle Monash Dataset Preparation to csv
    """
    def __init__(self, dataset : str):
        super().__init__()
        self.__dataset : dict = self._getConfig()["datasets"][dataset]

    def prepare(self, datasetConfig : dict) -> dict:
        """
        Method to prepare csv with monash dataset
        """
        separator : str = self.__dataset["separator"]
        decimal : str = self.__dataset["decimal"]
        period : int = TimeManager.getPeriodSeconds(self.__dataset["periodicity"])

        for subdataset in datasetConfig:
            timestamp : str = ""
            dataframeDict : dict = {
                "timestamp" : None,
            }
            maxLength : int = 0
            datasetLineFound = False
            with open(datasetConfig[subdataset], "r", encoding="utf-8", errors="replace") as datasetFile:
                while True:
                    nextLine : str = datasetFile.readline()
                    if nextLine is None:
                        break
                    elif nextLine[0:3] == "T1:":
                        datasetLineFound = True

                    if datasetLineFound:
                        if nextLine == "":
                            break
                        feature : str = nextLine.split(":")[0]
                        data : list = nextLine.split(f"{feature}:")[1].split(separator)
                        timestamp = data[0].split(":")[0]
                        samples : list = [
                            float(sample.replace(separator, ".")) for sample in data[1:]
                        ]
                        dataframeDict[feature] = pd.Series(samples)

                        if len(samples) > maxLength:
                            maxLength = len(samples)

            timestamps : list = []
            for _ in range(maxLength):
                timestamps.append(timestamp)
                timestamp = TimeManager.nextTimeStamp(
                    timestamp,
                    self._getConfig()["monash"]["timeFormat"],
                    period,
                )

            dataframeDict["timestamp"] = pd.Series(timestamps)

            dataframe: pd.core.frame.DataFrame = pd.DataFrame(dataframeDict)

            datasetConfig[subdataset] = datasetConfig[subdataset].split(".")[0] + ".csv"
            dataframe.to_csv(datasetConfig[subdataset], sep=separator, decimal=decimal, index=False)
            Utils.savePandasAsArrow(dataframe, datasetConfig[subdataset])

        return datasetConfig
