import os
import requests
from git import Repo
import kagglehub
from datasets import load_dataset
from utils.fileSystem import FileSystem
from exceptions.datasetException import DatasetException

class DatasetDownloader(FileSystem):
    """
    Class to handle datasets downloading
    """
    def __init__(self, datasetName : str, datasetParams : dict):
        super().__init__()
        self.__datasetParams : dict = datasetParams
        self.__datasetName : str = datasetName
        self.__typeDataset : str = datasetParams["type"]
        self.__datasetPath : str = self._createPath(
            self._getPaths()["datasets"],
            datasetName,
        )

    def __prepareGitRepository(self, url : str):
        """
        Method to prepare git repository
        """
        self._deleteFolder(self.__datasetPath)

        try:
            Repo.clone_from(url, self.__datasetPath)
        except Exception as e:
            raise DatasetException(
                f"Repository {url} could not be cloned, Error: {str(e)}"
            )

    def __downloadFile(self, url : str, path : str):
        """
        Method to download file
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP request errors
            with open(path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=8192):
                    zip_file.write(chunk)
        except requests.exceptions.RequestException as e:
            raise DatasetException(
                f"File {url} could not be downloaded, Error: {str(e)}"
            )

    def __downloadZip(self, url : str):
        """
        Method to download and unzip
        """
        zipFile : str = os.path.join(self.__datasetPath, self.__datasetName + ".zip")
        self._deleteFolder(self.__datasetPath)
        self._createFolder(self.__datasetPath)
        self.__downloadFile(url, zipFile)
        self._unzipFile(zipFile, self.__datasetPath)
        self._deleteFile(zipFile)

    def __downloadKaggle(self, url : str):
        """
        Method to download kaggle dataset
        """
        self._deleteFolder(self.__datasetPath)
        path : str = kagglehub.dataset_download(self.__datasetParams["datasetName"])
        self._moveFolder(path, self.__datasetPath)

    def downloadDataset(self) -> str:
        """
        Method to download dataset

        return str path where dataset is downloaded
        """
        if self.__typeDataset == "gitRepository":
            self.__prepareGitRepository(self.__datasetParams["url"])
        elif self.__typeDataset == "zip":
            self.__downloadZip(self.__datasetParams["url"])
        elif self.__typeDataset == "kaggle":
            self.__downloadKaggle(self.__datasetParams["url"])
        elif self.__typeDataset == "huggingface":
            huggingfaceDatasets : dict = {}
            downloadFolder : str = os.path.join(self.__datasetPath, "downloads")
            for subdataset in self.__datasetParams["subdatasets"]:
                huggingfaceDatasets[subdataset] = load_dataset(self.__datasetParams["datasetName"], subdataset, split="train", cache_dir=self.__datasetPath)
                if self._checkFileExists(downloadFolder):
                    self._deleteFolder(downloadFolder)
        else:
            raise DatasetException(
                f"Type {self.__typeDataset} on dataset {self.__datasetName} is not supported"
            )

        return self.__datasetPath
