import pandas as pd
import math
import struct
import lmdb
import json
import torch
from torch.utils.data import Dataset
import random
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from utils.fileSystem import FileSystem
from utils.utils import Utils
from datasetsModule.datasets import Datasets
from datasetsModule.datasetIterator import DatasetIterator
from model.moiraiMoe import MoiraiMoEEmbeddings

class DatasetTrainingIterator(Dataset):
    def __init__(
            self,
            config : dict,
            datasetsConfig : dict,
            balanced : bool = True,
            mapSize : int = 50 * 1024 * 1024 * 1024,  # 50GB
        ):
        self.__fileSystem : FileSystem = FileSystem()
        self.__config : dict = config
        self.__dataset : str = config["training"]["dataset"]
        self.__combinations : list = [f"{combination['contextLength']}_{combination['predictionLength']}" for combination in config["training"]["lengthCombinations"]]

        self.__databases : dict = {}
        self.__lmdbDatabasesConfig : dict = self.__loadConfig()
        self.__mapSize : int = mapSize
        self.__createLmdbPaths()
        random.seed(config["seed"])
        self.__batchSize : int = config["training"]["batchSize"]
        self.__maxIterationsPerEpoch : int = config["training"]["maxIterationsPerEpoch"]
        if (not balanced) or (not all([self.__isComplete(combination) for combination in self.__combinations])):
            Datasets().loadDataset(self.__dataset)
            self.__subdatasets = {key : None for key in list(datasetsConfig[self.__dataset].keys())}

            self.__iterators = self.getIterators(config["training"]["lengthCombinations"])
            if balanced:
                self.balanceDataset()

        if balanced:
            self.__balancedIterators : dict = {
                combination : self.__balancedIterator(combination, self.__config["training"]["batchSize"] * self.__config["training"]["maxIterationsPerEpoch"]) for combination in self.__combinations
            }

        self.__balanced : bool = balanced

    def __loadConfig(self) -> dict:
        """
        Method to load dataset config
        """
        datasetConfig : dict = Utils.readYaml(
            self.__fileSystem._getFiles()["lmdbDatabasesConfig"]
        )
        return datasetConfig if type(datasetConfig) == dict else dict()

    def __writeConfig(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self.__fileSystem._getFiles()["lmdbDatabasesConfig"],
            self.__loadConfig() | entry,
        )

    def __createLmdbPaths(self):
        """
        Create the LMDB for the dataset.
        This method creates an LMDB for the dataset, which is used to store the samples.
        """
        indeces : list = []
        for combination in self.__combinations:
            if combination not in self.__lmdbDatabasesConfig:
                self.__lmdbDatabasesConfig[combination] = {
                    "database": Utils.appendPath(self.__fileSystem._getPaths()["lmdbDatabasePath"], [f"{self.__dataset}-{combination}.lmdb"]),
                    "subdatasets": [],
                }

        self.__writeConfig(self.__lmdbDatabasesConfig)

    def __getLdmbDatabase(self, combination : str) -> lmdb.Environment:
        """
        Open the LMDB database for a given combination.
        This method opens the LMDB database for the specified combination and returns the environment.
        """
        lmdbDatabase : lmdb.Environment = lmdb.open(
            self.__lmdbDatabasesConfig[combination]["database"],
            map_size=self.__mapSize,
            max_dbs=1,
        )
        with lmdbDatabase.begin(write=True) as txn:
            metadata : dict = txn.get(b'__metadata__')
            if metadata is None:
                txn.put(b'__metadata__', json.dumps({"count" : 0, "complete" : False}).encode())

        return lmdbDatabase

    def __getDatabaseIndexRange(self, combination : str) -> tuple[float, float, int]:
        """
        Get the range of keys in the LMDB database for a given combination.
        This method retrieves the minimum and maximum keys from the LMDB database for the specified combination.

        returns:
        tuple: A tuple containing the minimum key, maximum key, and the number of samples in the database.
        """
        minList : list = []
        maxList : list = []
        count : int = 0
        lmdbDatabase : lmdb.Environment = self.__getLdmbDatabase(combination)
        with lmdbDatabase.begin(write=False) as txn:
            cursor = txn.cursor()
            if cursor.first():
                minList.append(self.__sortKeyToFloat(cursor.key()))

            if cursor.last():
                maxList.append(self.__sortKeyToFloat(cursor.key()))

            count = json.loads(txn.get(b'__metadata__')).get("count", 0)

        lmdbDatabase.close()

        return min(minList), max(maxList), count

    def __floatToSortKey(self, value: float) -> bytes:
        packed : bytes = struct.pack('>d', value)
        i = struct.unpack('>Q', packed)[0]  # interpret bits as unsigned int
        if value < 0:
            i = ~i & 0xFFFFFFFFFFFFFFFF  # bitwise NOT for negative values
        else:
            i ^= (1 << 63)  # flip sign bit for positive

        return struct.pack('>Q', i)

    def __sortKeyToFloat(self, b: bytes) -> float:
        if not isinstance(b, bytes):
            return
    
        if len(b) != 8:
            return
        i = struct.unpack('>Q', b)[0]

        if i & (1 << 63):
            i ^= (1 << 63)  # unflip sign bit
        else:
            i = ~i & 0xFFFFFFFFFFFFFFFF  # reverse bitwise NOT

        return struct.unpack('>d', struct.pack('>Q', i))[0]

    def __ingestSample(self, key : float, sample : dict, combination : str):
        """
        Ingest a sample into the LMDB database.
        This method stores the sample in the LMDB database.
        """
        lmdbDatabase : lmdb.Environment = self.__getLdmbDatabase(combination)
        
        try:
            with lmdbDatabase.begin(write=True) as txn:
                count : int = json.loads(txn.get(b'__metadata__')).get("count", 0)
                existing : bytes = txn.get(self.__floatToSortKey(key))
                if existing is None:
                    count += 1
                    txn.put(self.__floatToSortKey(key), json.dumps(sample).encode())
                    txn.put(b'__metadata__', json.dumps({"count" : count}).encode())
        except lmdb.Error as e:
            pass

        lmdbDatabase.close()
        del lmdbDatabase

    def __iterateSamples(self, combination : str) -> list:
        """
        Iterate over the samples in the LMDB database.
        This method retrieves all samples from the LMDB database for a given combination.
        """
        with self.__databases[combination].begin(write=False) as txn:
            cursor = txn.cursor()
            if cursor.first():
                for key, value in cursor:
                    if key.startswith(b'__'):
                        continue
                    sample : dict = json.loads(value.decode())
                    key : float = struct.unpack('>d', key)[0]
                    yield key, sample

    def __setComplete(self, combination : str, complete : bool = True):
        """
        Set the completion status of the dataset.
        This method updates the metadata in the LMDB database to indicate whether the dataset is complete.
        """
        lmdbDatabase : lmdb.Environment = self.__getLdmbDatabase(combination)
        with lmdbDatabase.begin(write=True) as txn:
            metadata : dict = json.loads(txn.get(b'__metadata__'))
            metadata["complete"] = True
            txn.put(b'__metadata__', json.dumps(metadata).encode())

        lmdbDatabase.close()

    def __isComplete(self, combination : str) -> bool:
        """
        Check if the dataset is complete.
        This method checks the metadata in the LMDB database to determine if the dataset is complete.
        """
        lmdbDatabase : lmdb.Environment = self.__getLdmbDatabase(combination)
        with lmdbDatabase.begin(write=False) as txn:
            metadata : dict = json.loads(txn.get(b'__metadata__'))
            return metadata.get("complete", False)

        lmdbDatabase.close()

    def __resetIterators(self):
        """
        Reset the iterators for the dataset.
        """
        for key in list(self.__iterators.keys()):
            for element in self.__subdatasets.keys():
                self.__iterators[key].resetIteration(element, True, trainPartition=self.__config["trainPartition"])

    def getIterators(self, combinations : list) -> dict:
        """
        Get the iterators for the dataset.
        """
        iterators : dict = {}
        for combination in combinations:
            contextLength : int = combination["contextLength"]
            predictionLength : int = combination["predictionLength"]
            iterator : DatasetIterator = Datasets().loadDataset(self.__dataset)
            iterator.setSampleSize(contextLength + predictionLength)
            for element in self.__subdatasets.keys():
                if self.__subdatasets[element] is None:
                    self.__subdatasets[element] = list(iterator.getAvailableFeatures(element).keys())
                iterator.resetIteration(element, True, trainPartition=self.__config["trainPartition"])
            iterators[f"{contextLength}_{predictionLength}"] = iterator
        return iterators

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, int]:
        """
        Get a random sample from the dataset.
        This method retrieves a random sample from the dataset, ensuring that the sample is valid and does not contain any NaN values.
        """
        if self.__balanced:
            return self.__getitem__balanced(idx)
        else:
            return self.__getitem__raw(idx)

    def test(self, idx : int) -> tuple[torch.Tensor, int]:
        return self.__getitem__(idx)

    def __getitem__balanced(self, idx : int) -> tuple[torch.Tensor, int]:
        """
        Get a random sample from the dataset.
        This method retrieves a random sample from the dataset, ensuring that the sample is valid and does not contain any NaN values.
        """
        samples : torch.Tensor = None  

        running : bool = True
        batchIndex : int = 0

        combination : str = random.choice(self.__combinations)
        contextLength : int = int(combination.split("_")[0])
        predictionLength : int = int(combination.split("_")[1])

        if idx == 0:
            self.__balancedIterators : dict = {
                combination : self.__balancedIterator(combination, self.__config["training"]["batchSize"] * self.__config["training"]["maxIterationsPerEpoch"]) for combination in self.__combinations
            }

        while running:
            sample : list = next(self.__balancedIterators[combination], None)
            if sample is None:
                running = False
                break

            torchSample : torch.Tensor = torch.tensor(
                sample,
                dtype=torch.float32,
            ).unsqueeze(0)

            if batchIndex >= self.__batchSize:
                running = False
            else:
                if samples is None:
                    samples = torchSample
                else:
                    samples = torch.cat((samples, torchSample), dim=0)

            batchIndex += 1

        return samples, contextLength

    def __getitem__raw(self, idx : int) -> tuple[torch.Tensor, int]:
        """
        Get a random sample from the dataset.
        This method retrieves a random sample from the dataset, ensuring that the sample is valid and does not contain any NaN values.
        """
        samples : torch.Tensor = None  

        running : bool = True
        batchIndex : int = 0

        combination : str = random.choice(list(self.__iterators.keys()))
        iterator : DatasetIterator = self.__iterators[combination]
        contextLength : int = int(combination.split("_")[0])
        predictionLength : int = int(combination.split("_")[1])

        if idx == 0:
            self.__resetIterators()

        while running:
            subdataset : str = random.choice(list(self.__subdatasets.keys()))
            features : list = self.__subdatasets[subdataset]
            try:
                sample : pd.core.frame.DataFrame = iterator.iterateDataset(
                    subdataset,
                    self.__subdatasets[subdataset],
                    True,
                )
                if sample is None:
                    continue
                if len(sample) < predictionLength + contextLength:
                    continue

                indexes : list = [index for index in range(1,len(features))]
                random.shuffle(indexes)
                for i in range(len(indexes)):
                    index : int = indexes[i]
                    if sample[index].isna().any().any():
                        continue

                    torchSample : torch.Tensor = torch.tensor(
                        sample[[index]].to_numpy(),
                        dtype=torch.float32,
                    ).permute(1,0)

                    mean = torchSample.mean()
                    if torch.all(torch.abs(torchSample - mean) < 0.1):
                        continue

                    if (torchSample == 0.0).float().mean() > 0.75:
                        continue

                    if batchIndex >= self.__batchSize:
                        running = False
                    else:
                        if samples is None:
                            samples = torchSample
                        else:
                            samples = torch.cat((samples, torchSample), dim=0)

                    batchIndex += 1
            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        return samples, contextLength

    def __iterator(self, combination : str, subdataset : str, features : list = [], train : bool = True) -> pd.core.frame.DataFrame:
        """
        Iterate through the dataset.
        This method retrieves a sample from the dataset for training or testing.
        """
        while True:
            try:
                sample : pd.core.frame.DataFrame = self.__iterators[combination].iterateDataset(
                    subdataset,
                    features,
                    train
                )
                if sample is not None:
                    yield sample
                else:
                    self.__iterators[combination].releaseIterator(subdataset)
                    return
            except Exception as e:
                continue

    def __balancedIterator(self, combination : str, nSamples : int) -> pd.core.frame.DataFrame:
        """
        Iterate through the dataset in a balanced way.
        This method retrieves a sample from the dataset for training or testing, ensuring that the dataset is balanced.
        """
        min, max, count = self.__getDatabaseIndexRange(combination)
        step : int = math.floor((max - min) / (nSamples * len(self.__combinations)))
        databaseIndex : float = min

        lmdbDatabase : lmdb.Environment = self.__getLdmbDatabase(combination)
        count = 0
        with lmdbDatabase.begin(write=False) as txn:
            cursor = txn.cursor()
            if cursor.first():
                while True:
                    if not cursor.next():
                        return
                    else:
                        currentKey : float = self.__sortKeyToFloat(cursor.key())
                        if currentKey is None:
                            count += 1
                            print("Skipping invalid key: " + str(cursor.key()))
                            continue
                        if currentKey >= databaseIndex:
                            yield json.loads(cursor.value().decode())["sample"]
                        databaseIndex += step

        lmdbDatabase.close()

    def balanceDataset(self):
        """
        Balance dataset to ensure the dataset is diversed
        """
        self.__databasesIndex : int = 0
        maxSamples : int = self.__config["training"]["maxSamplesPerSubdataset"]
        for combination in self.__combinations:
            contextLength : int = int(combination.split("_")[0])
            predictionLength : int = int(combination.split("_")[1])

            model : MoiraiMoEForecast = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-small"),
                prediction_length=predictionLength,
                context_length=contextLength+predictionLength,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            moiraiMoEEmbeddings : MoiraiMoEEmbeddings = MoiraiMoEEmbeddings(model.module)

            for subdataset in list(self.__subdatasets.keys()):
                sampleIndex : int = 0
                iterator = self.__iterator(combination, subdataset, self.__subdatasets[subdataset], True)
                print("ingesting dataset: " + subdataset + " combination: " + combination)
                if subdataset in self.__lmdbDatabasesConfig[combination]["subdatasets"]:
                    continue
                running : bool = True
                while running:
                    features : list = self.__subdatasets[subdataset]
                    sample : pd.core.frame.DataFrame = next(iterator, None)
                    if sample is None:
                        running = False
                        break
                    if len(sample) < predictionLength + contextLength:
                        continue

                    indexes : list = [index for index in range(1,len(features))]
                    for i in range(len(indexes)):
                        index : int = indexes[i]
                        if sample[index].isna().any().any():
                            continue

                        if sampleIndex >= maxSamples:
                            running = False
                            break

                        torchSample : torch.Tensor = torch.tensor(
                            sample[[index]].to_numpy(),
                            dtype=torch.float32,
                        ).permute(1,0).squeeze(0)

                        mean = torchSample.mean()
                        if torch.all(torch.abs(torchSample - mean) < 0.1):
                            continue

                        if (torchSample == 0.0).float().mean() > 0.75:
                            continue

                        output = moiraiMoEEmbeddings.forward(
                            torchSample.numpy(),
                        ).squeeze(0)

                        self.__ingestSample(
                            output.sum().item(),  # Use the sum of the output as the key for LMDB
                            {
                                "sample": torchSample.clone().numpy().tolist(),
                            },
                            combination,
                        )
                        del mean
                        del output
                        del torchSample
                        del index

                        sampleIndex += 1

                self.__lmdbDatabasesConfig[combination]["subdatasets"].append(subdataset)
                self.__writeConfig(self.__lmdbDatabasesConfig)
                del iterator
            del model
            del moiraiMoEEmbeddings
            self.__setComplete(combination, True)

    def __len__(self):
        return self.__maxIterationsPerEpoch
