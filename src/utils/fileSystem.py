import os
import shutil
import zipfile
from pathlib import Path
from utils.utils import Utils

class FileSystem(object):
    """
    Class to manage file system
    """
    def __init__(self) -> None:
        self.__rootPath : str = Path(os.path.abspath(__file__)).parents[1]
        self.__config : dict = Utils.readYaml(
            os.path.join(self.__rootPath, "config.yaml")
        )
        self.__paths : dict = self.__loadPaths()
        self.__files : dict = self.__loadFiles()

    def __loadPaths(self) -> dict:
        """
        Private method to load paths from config
        """
        paths : dict = dict()

        for path in self.__config["paths"]:
            longPath : str = Utils.appendPath(self.__rootPath, self.__config["paths"][path])
            paths.update({
                path : longPath,
            })
            os.makedirs(longPath, exist_ok=True)

        return paths

    def __loadFiles(self) -> dict:
        """
        Private method to load files from config
        """
        paths : dict = dict()

        for path in self.__config["files"]:
            longPath : str = Utils.appendPath(self.__rootPath, self.__config["files"][path])
            paths.update({
                path : longPath,
            })
            if not os.path.exists(longPath):
                open(longPath, 'a').close()

        return paths

    def _deleteFolder(self, folder : str):
        """
        Private method to delete a folder
        """
        if not os.path.exists(folder):
            raise Exception("Folder " + folder + " does not exists")
        else:
            shutil.rmtree(folder)

    def _checkFileExists(self, filePath : str) -> bool:
        """
        Private method to check if file exists
        """
        return os.path.exists(filePath)
    
    def _moveFolder(self, folderPath : str, destinationPath : str):
        """
        Private method to move folder from one location to other
        """
        shutil.move(folderPath, destinationPath)

    def _deleteFileFromFiles(self, fileKey : str):
        """
        Protected Method to delete a file
        """
        if fileKey not in self.__files:
            raise Exception("File " + fileKey + " does not exists")
        else:
            self._deleteFile(self.__files[fileKey])

    def _deleteFile(self, filePath : str):
        """
        Protected Method to delete a file
        """
        os.remove(filePath)

    def _deleteFolderContent(self, folderKey : str):
        """
        Protected Method to delete content of a folder
        """
        if folderKey not in self.__paths:
            raise Exception("Folder " + folderKey + " does not exists")
        else:
            self._deleteFolder(self.__paths[folderKey])
            self._createFolder(self.__paths[folderKey])

    def _getFiles(self) -> dict:
        """
        Public method to get files
        """
        return self.__files

    def _getPaths(self) -> dict:
        """
        Public method to get paths
        """
        return self.__paths

    def _getConfig(self) -> dict:
        """
        Private method to get config
        """
        return self.__config

    def _createPath(self, *subdicts : str) -> str:
        """
        Method to create a path based on a list of subdicts
        """
        path : dict = os.path.join(*subdicts)
        self._createFolder(path)

        return path

    def _createFolder(self, folderName : str):
        """
        Public method to create a folder
        """
        os.makedirs(folderName, exist_ok=True)

    def _unzipFile(self, zipFile : str, path : str, noFolders : bool = False):
        """
        Method to unzip file
        """
        if noFolders:
            with zipfile.ZipFile(zipFile, 'r') as zipRef:
                for file in zipRef.namelist():
                    if not file.endswith('/'):
                        fileName = os.path.basename(file)

                        with zipRef.open(file) as source, open(os.path.join(path, fileName), 'wb') as target:
                            target.write(source.read())
        else:
            with zipfile.ZipFile(zipFile, "r") as zipRef:
                zipRef.extractall(path)

    def _saveFile(self, fileName : str, content : list):
        """
        Method to save content in file
        """
        with open(fileName, "w") as file:
            file.writelines(content)