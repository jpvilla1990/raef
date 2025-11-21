import os
from pathlib import Path
import pandas as pd
from utils.fileSystem import FileSystem
from utils.utils import Utils
from exceptions.datasetException import DatasetException
from datasetsModule.datasetDownloader import DatasetDownloader
from datasetsModule.datasetIterator import DatasetIterator
from datasetsModule.huggingFaceIterator import HuggingFaceIterator
from datasetsModule.monashPreparer import MonashPreparer
from utils.utils import Utils

class Datasets(FileSystem):
    """
    Class to load datasets
    """
    def __init__(self):
        super().__init__()
        self.__datasets : dict = self._getConfig()["datasets"]

    def __loadDatasetConfig(self) -> dict:
        """
        Method to load dataset config
        """
        datasetConfig : dict = Utils.readYaml(
            self._getFiles()["datasets"]
        )
        return datasetConfig if type(datasetConfig) == dict else dict()

    def __writeDatasetConfig(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()["datasets"],
            self.__loadDatasetConfig() | entry,
        )

    def __verifyDatasetIsDownloaded(self, dataset : str) -> bool:
        """
        Method to verify if dataset is already downloaded
        """
        if dataset in self.__loadDatasetConfig():
            return True
        else:
            return False

    def loadDataset(self, dataset : str, forceDownload : bool = False):
        """
        Method to load dataset
        """
        if dataset not in self.__datasets:
            raise DatasetException(f"Dataset {dataset} does not exists")

        if not self.__verifyDatasetIsDownloaded(dataset) or forceDownload:
            datasetDownloader : DatasetDownloader = DatasetDownloader(
                dataset,
                self.__datasets[dataset],
            )
            datasetPath : str = datasetDownloader.downloadDataset()
            datasetConfig : dict = {
                subDataset: os.path.join(
                    datasetPath,
                    *self.__datasets[dataset]["subdatasets"][subDataset]
                ) for subDataset in self.__datasets[dataset]["subdatasets"]
            }

            datasetFormat : str = self.__datasets[dataset]["format"]
            if datasetFormat == "csv":
                for subDataset in datasetConfig:
                    subDatasetPath : str = datasetConfig[subDataset]
                    if ".zip" in subDatasetPath:
                        self._unzipFile(subDatasetPath, Path(subDatasetPath).parent, True)
                        subDatasetPath = subDatasetPath.replace(".zip", ".csv")
                        datasetConfig[subDataset] = subDatasetPath
                    df : pd.core.frame.DataFrame = pd.read_csv(
                        subDatasetPath,
                        sep=self.__datasets[dataset]["separator"],
                        decimal=self.__datasets[dataset]["decimal"],
                    )
                    Utils.savePandasAsArrow(df, subDatasetPath)
            elif datasetFormat == "monash":
                monashPreparer : MonashPreparer = MonashPreparer(dataset)
                datasetConfig = monashPreparer.prepare(datasetConfig)
            elif datasetFormat == "huggingface":
                pass
            else:
                raise DatasetException(f"Format {datasetFormat} is not supported in dataset {dataset}")

            self.__writeDatasetConfig(
                {dataset : datasetConfig}
            )

        if self.__datasets[dataset]["format"] == "huggingface":
            return HuggingFaceIterator(
                dataset,
                self.__loadDatasetConfig()[dataset],
                self._createPath(
                    self._getPaths()["datasets"],
                    dataset,
                ),
                self.__datasets[dataset],
                self._getConfig()["seed"],
                self._getConfig()["maxNan"],
                self._getConfig()["minUniqueElementRatio"],
            )
        else:
            return DatasetIterator(
                dataset,
                self.__loadDatasetConfig()[dataset],
                self.__datasets[dataset],
                self._getConfig()["seed"],
            )