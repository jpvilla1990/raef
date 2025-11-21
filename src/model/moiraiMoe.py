import uuid
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from jaxtyping import Bool, Float, Int

from uni2ts.eval_util.plot import plot_single

from gluonts.torch import PyTorchPredictor
import uni2ts
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast

from utils.timeManager import TimeManager
from utils.fileSystem import FileSystem
from utils.utils import Utils

from vectorDB.vectorDB import vectorDB
from exceptions.modelException import ModelException
from model.ragCrossAttention import RagCrossAttention

class MoiraiMoEEmbeddings(nn.Module):
    def __init__(self, moiraRaiModule : MoiraiMoEModule):
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Extracting layers from the original model including first normalization layer before attention module
        self.scaler : uni2ts.module.packed_scaler.PackedStdScaler = moiraRaiModule.scaler
        self.inProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.in_proj.to(self.__device)
        self.resProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.res_proj.to(self.__device)
        self.featProj : uni2ts.module.ts_embed.FeatLinear = moiraRaiModule.feat_proj.to(self.__device)
        self.norm : uni2ts.module.norm.RMSNorm = moiraRaiModule.encoder.layers[0].norm1.to(self.__device)

    def forward(
            self,
            x : np.ndarray,
            patchSize : int = 16,
            batchSize : int = 1,
        ) -> torch.Tensor:
        """
        Define the forward pass based on the selected layers.
        Assuming:
        - `scaler` normalizes input.
        - `in_proj`, `res_proj`, and `feat_proj` apply transformations.
        """
        seqLen : int = math.ceil(len(x) / patchSize) + 1
        pad : np.ndarray = np.zeros((batchSize * (seqLen - 1) * patchSize) - len(x))
        if len(pad) > 0:
            x = np.concatenate([pad, x])
        patchSizeTensor : torch.Tensor = torch.full((batchSize, seqLen), patchSize).to(self.__device)
        target : torch.Tensor = F.pad(
            torch.tensor(x, dtype=torch.float32).reshape(batchSize, seqLen - 1, patchSize),
            (0, 0, 0, 1),
            value=0,
        ).to(self.__device)
        observedMask : torch.Tensor = torch.ones((batchSize, seqLen, patchSize), dtype=torch.bool).to(self.__device)
        predictionMask : torch.Tensor = torch.zeros((batchSize, seqLen), dtype=torch.bool).to(self.__device)
        predictionMask[0][:][-1] = True
        sampleId : torch.Tensor = torch.ones((batchSize, seqLen), dtype=torch.int32).to(self.__device)
        variateId : torch.Tensor = torch.zeros((batchSize, seqLen), dtype=torch.int32).to(self.__device)

        loc, scale = self.scaler(
            target,
            observedMask * ~predictionMask.unsqueeze(-1),
            sampleId,
            variateId,
        )
        scaledTarget : torch.Tensor = (target - loc) / scale
        scaledTarget = scaledTarget.to(self.__device)
        inRepr : torch.Tensor = self.inProj(scaledTarget, patchSizeTensor)
        inRepr : torch.Tensor = F.silu(inRepr)
        inRepr : torch.Tensor = self.featProj(inRepr, patchSizeTensor)
        resRepr : torch.Tensor = self.resProj(scaledTarget, patchSizeTensor)

        # Combine or return outputs depending on your use case
        return self.norm(inRepr + resRepr)

    def inference(
            self,
            x : np.ndarray,
            patchSize : int = 16,
            batchSize : int = 1,
        ) -> torch.Tensor:
        """
        Inference
        """
        output : torch.Tensor = None
        with torch.no_grad():
            output = self(x, patchSize, batchSize)

        return output.to("cpu")

class MoiraiMoE(FileSystem):
    """
    Class to handle MoiraiMoe
    """
    def __init__(
        self,
        modelSize : str = "small",
        predictionLength : int = 20,
        contextLength : int = 200,
        patchSize : int = 16,
        numSamples : int = 100,
        targetDim : int = 1,
        featDynamicRealDim : int = 0,
        pastFeatDynamicRealDim : int = 0,
        batchSize : int = 1,
        createPredictor : bool = True,
        frozen : bool = True,
        loadPretrainedModel : bool = False,
        loadFineTunedModel : bool = False,
        fineTunedModel : str = "",
    ):
        super().__init__()
        self.__patchSize : int = patchSize
        self.__datasetsConfig : dict = self._getConfig()["datasets"]
        self.__timestampFormat : str = "%d-%m-%Y %H:%M:%S"
        self.__timeDiffsSeconds : dict = {
            "S" : 1,
            "T" : 60,
            "H" : 60 * 60,
            "D" : 60 * 60 * 24,
            "W" : 60 * 60 * 24 * 7,
            "M" : 60 * 60 * 24 * 30,
            "Y" : 60 * 60 * 24 * 365,
        }

        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__scoreThreshold : float = self._getConfig()["vectorDatabase"]["scoreThreshold"]
        self.__k : int = self._getConfig()["vectorDatabase"]["k"]
        self.__numberSamples : int = numSamples

        self.__contextLength : int = contextLength
        self.__predictionLength : int = predictionLength
        self.model : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length=contextLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        if loadFineTunedModel:
            checkpoint : dict = torch.load(os.path.join(self._getPaths()["pretrainedModels"], fineTunedModel), weights_only=False)
            parsedCheckpoint : dict = {}
            for key in checkpoint["state_dict"].keys():
                if key.startswith("model.module."):
                    parsedCheckpoint[key.replace("model.module.", "")] = checkpoint["state_dict"][key]
                else:
                    parsedCheckpoint[key] = checkpoint["state_dict"][key]
            self.model.module.load_state_dict(
                parsedCheckpoint,
            )

        self.modelRag : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length= contextLength + predictionLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.modelRagCA : RagCrossAttention = RagCrossAttention(
            patchSize=self.__patchSize,
            pretrainedModel=self._getFiles()["paramsRagCA"],
            loadPretrainedModel=loadPretrainedModel,
        )

        self.modelRagCABackBone : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length= contextLength + contextLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.__moiraiMoEEmbeddings : MoiraiMoEEmbeddings = MoiraiMoEEmbeddings(self.model.module)
        self.__vectorDB : vectorDB = vectorDB()

        if createPredictor:
            self.__predictor : PyTorchPredictor = self.model.create_predictor(batch_size=batchSize)
            self.__predictorRag : PyTorchPredictor = self.modelRag.create_predictor(batch_size=batchSize)

        if frozen:
            for param in self.model.module.parameters():
                param.requires_grad = False
            for param in self.modelRag.module.parameters():
                param.requires_grad = False
            for param in self.modelRagCABackBone.module.parameters():
                param.requires_grad = False

        self.model.module = self.model.module.to(self.__device)
        self.modelRag.module = self.modelRag.module.to(self.__device)
        self.modelRagCABackBone.module = self.modelRagCABackBone.module.to(self.__device)

    def __patching(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply patching to the input tensor.
        This method reshapes the input tensor into patches of a specified size.
        :param x: Input tensor to be patched.
        :return: Patches of the input tensor.
        """
        seqLength = x.shape[1]
        remainder : int  = (seqLength - self.__patchSize) % self.__patchSize
        padLen : int = self.__patchSize - remainder if remainder != 0 else 0
        x = F.pad(x, (0, padLen), value=0)
        x = x.unfold(dimension=1, size=self.__patchSize, step=self.__patchSize)
        zeros : torch.Tensor = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        return torch.cat([zeros, x, zeros], dim=1)

    def forwardRagCA(self, x : torch.Tensor, moiraiMoEOnly : bool = False) -> torch.Tensor:
        """
        Forward pass RAG Cross Attention
        """
        device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target : torch.Tensor = self.__patching(x)

        batchSize : int = target.shape[0]
        numPatches : int = target.shape[1]
        patchSizeTensor : torch.Tensor = torch.full(
            (batchSize, numPatches),
            self.__patchSize,
            device=device,
        )
        observedMask : torch.Tensor = torch.ones(
            (batchSize, numPatches, self.__patchSize),
            dtype=torch.bool,
            device=device,
        )
        observedMask[:,0,:] = False
        predictionMask : torch.Tensor = torch.zeros(
            (batchSize, numPatches),
            dtype=torch.bool,
            device=device,
        )
        predictionMask[:,-1] = True
        sampleId : torch.Tensor = torch.ones(
            (batchSize, numPatches),
            dtype=torch.int32,
            device=device,
        )
        sampleId[:,0] = 0
        timeId : torch.Tensor = torch.arange(
            numPatches - 1,
            device=device,
        )
        timeId = timeId.unsqueeze(0)
        timeId = timeId.repeat(batchSize, 1)
        timeId = F.pad(timeId, (1, 0), value=0)

        variateId : torch.Tensor = torch.zeros(
            (batchSize, numPatches),
            device=device,
            dtype=torch.int64,
        )

        if moiraiMoEOnly:
            return self.model.module(
                target=target,
                observed_mask=observedMask,
                sample_id=sampleId,
                time_id=timeId,
                variate_id=variateId,
                prediction_mask=predictionMask,
                patch_size=patchSizeTensor,
            )
        else:
            return self.modelRagCABackBone.module(
                target=target,
                observed_mask=observedMask,
                sample_id=sampleId,
                time_id=timeId,
                variate_id=variateId,
                prediction_mask=predictionMask,
                patch_size=patchSizeTensor,
            )

    def setRagCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAG collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            self.__moiraiMoEEmbeddings.inference,
        )

    def setRafCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAF collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            lambda x : torch.tensor(x).reshape(1, len(x)),
        )

    def setRafCosCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAF collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            lambda x : (torch.tensor(x).reshape(1, len(x)) - torch.mean(torch.tensor(x).reshape(1, len(x)))) / (torch.std(torch.tensor(x).reshape(1, len(x))) + 1e-8),
        )

    def __getFrequency(self, timestamps : pd.core.frame.DataFrame, timestampFormat : str) -> str:
        """
        Method to get frequency from the timestamps
        """
        timeDiffSeconds : int = TimeManager.timeDiffSeconds(timestamps.iloc[0], timestamps.iloc[1], timestampFormat)

        distances : dict = {abs(self.__timeDiffsSeconds[key] - timeDiffSeconds) : key for key in self.__timeDiffsSeconds}

        return distances[sorted(distances.keys())[0]]

    def ingestVector(self, sample : np.ndarray, prediction : np.ndarray, dataset : str = ""):
        """
        Method to ingest vector
        """
        self.__vectorDB.ingestTimeseries(sample, prediction, dataset)

    def deleteDataset(self, dataset : str):
        """
        Method to delete dataset from collection
        """
        self.__vectorDB.deleteDataset(dataset)

    def deleteCollection(self, collectionName : str, dataset : str):
        """
        Method to delete a collection
        """
        self.__vectorDB.deleteCollection(collectionName, dataset)

    def queryVector(self, sample : np.ndarray, k : int = 1) -> tuple:
        """
        Method to query vector
        """
        return self.__vectorDB.queryTimeseries(sample, k)

    def queryBatchVector(self, batch : torch.Tensor, k : int = 1) -> tuple:
        """
        Method to query vector
        """
        queriedBatch : torch.Tensor = None
        scoreBatch : torch.Tensor = None
        for element in batch:
            queried, score = self.__vectorDB.queryTimeseries(element, k)
            queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
            scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)
            if queriedBatch is None:
                queriedBatch = queriedTorch
                scoreBatch = scoreTensor
            else:
                queriedBatch = torch.cat((queriedBatch, queriedTorch), dim=0)
                scoreBatch = torch.cat((scoreBatch, scoreTensor), dim=0)

        return queriedBatch, scoreBatch

    def mergeQueries(self, query : tuple) -> np.ndarray:
        """
        Method to merge queries
        """
        if len(query[0]) == 0:
            return None
        merged : np.ndarray = np.zeros_like(query[0][0])
        nElements : int = 0
        for index in range(len(query[0])):
            if query[1][index] > self.__scoreThreshold:
                merged += query[0][index]
                nElements += 1

        if nElements == 0:
            return None
        else:
            return merged / nElements

    def mergeQueriesSoftMax(self, query : tuple, cosine : bool) -> np.ndarray:
        """
        Method to merge queries
        """
        vectors : np.ndarray = np.array(query[0])
        scores : np.ndarray = np.array(query[1]) if cosine else np.array(query[1])/(np.std(np.array(query[1])) + 1e-8)

        validIndices : np.ndarray = scores > self.__scoreThreshold if cosine else scores > 0 # If score is L2, then the score does not makes sense and just filters negative L2

        validScores : np.ndarray = scores[validIndices]
        validVectors : np.ndarray = vectors[validIndices]

        softMaxNumerator : np.ndarray = np.exp(validScores)
        weights : np.ndarray = softMaxNumerator / (softMaxNumerator.sum() + 1e-8)

        weigthedVectors : np.ndarray = validVectors * weights[:, np.newaxis]

        if len(validScores) == 0:
            return None
        else:
            return np.sum(weigthedVectors, axis=0)

    def inference(self, sample : pd.core.frame.DataFrame, dataset : str) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        sampleGluonts : ListDataset = ListDataset(
            [{
                "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                "target": sample["value"].tolist(),
            }],
            freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
        )
        #import time
        #start = time.perf_counter()
        prediction = next(iter(self.__predictor.predict(sampleGluonts)))

        #end = time.perf_counter()

        #print(f"Inference time: {(end - start) * 1000:.3f} ms")
        return prediction.quantile(0.5)

    def ragInference(
            self,
            sample : pd.core.frame.DataFrame,
            dataset : str,
            softMax : bool = False,
            cosine : bool = True,
            ragPredOnly : bool = False,
            plot : bool = False,
        ) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k)
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            sampleNp : np.ndarray = sample["value"].to_numpy()
            queriedMean, queriedStd = np.mean(queried), np.std(queried)
            sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

            queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
            sampleNormed : np.ndarray = (sampleNp - sampleMean) / (sampleStd + 1e-8)

            newSample : list = queryNormed.tolist() + sampleNormed.tolist()
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": newSample,
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            predictionNormed : np.ndarray = next(iter(self.__predictorRag.predict(sampleGluonts))).quantile(0.5)
            prediction : np.ndarray = (predictionNormed * (sampleStd + 1e-8)) + sampleMean

            if plot:
                Utils.plot(
                    [query.tolist() for query in queriedVectors[0]],
                    "rag.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [queried.tolist()],
                    "ragAvg.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [newSample + predictionNormed.tolist()],
                    "inputAndRag.png",
                    ":",
                    self.__contextLength,
                    rag=True,
                )
                Utils.plot(
                    [sample["value"].tolist() + prediction.tolist()],
                    "pred.png",
                    "-",
                    self.__contextLength,
                )

            return prediction
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def ragCaInference(
            self,
            sample : pd.core.frame.DataFrame,
            dataset : str,
            cosine : bool = True,
            plot : bool = False,
            extended : bool = False,
        ) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        query : torch.tensor = torch.tensor(sample["value"].to_numpy(), dtype=torch.float32)
        queried, score = self.queryVector(query, k=self.__k)
        if queried is not None:
            xContext : torch.Tensor = query.unsqueeze(0)
            queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
            scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)

            augmentedSample, mean, std = self.modelRagCA.inference(
                xContext,
                queriedTorch,
                scoreTensor,
                extended,
            )

            steps : int = math.ceil(self.__predictionLength / self.__patchSize)

            prediction : np.ndarray = np.empty((augmentedSample.shape[0], 0))
            timeSeries : np.ndarray = augmentedSample
            augmentedContextLength : int = augmentedSample.shape[1]

            for step in range(steps):
                pred = self.forwardRagCA(
                    timeSeries,
                    True,
                )

                pred = pred.sample(torch.Size((self.__numberSamples,))).median(dim=0).values[:,-2,:].clone()

                predNp : np.ndarray = pred.to("cpu").numpy()

                prediction = np.concatenate((prediction, predNp), axis=1)
                timeSeries = torch.cat((timeSeries[:, self.__patchSize:], pred), dim=1)

            stdNp : np.ndarray = std.to("cpu").squeeze(-1).squeeze(-1).numpy()
            meanNp : np.ndarray = mean.to("cpu").squeeze(-1).squeeze(-1).numpy()
            prediction = (prediction * stdNp) + meanNp

            prediction = prediction.squeeze(0)

            #newSample : list = augmentedSample[0].tolist()
            #sampleGluonts : ListDataset = ListDataset(
            #    [{
            #        "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
            #        "target": newSample,
            #    }],
            #    freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            #)
            #prediction : np.ndarray = next(iter(self.__predictorRag.predict(sampleGluonts))).mean
            id : str = str(uuid.uuid4())
            if plot:
                Utils.plot(
                    [query.tolist() for query in queried],
                    "images/rag" + id + ".png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [augmentedSample[0].tolist()],
                    "images/augmentedRa" + id + ".png",
                )
                Utils.plot(
                    [sample["value"].tolist() + prediction.tolist()],
                    "images/pred.png" + id + ".png",
                    "-",
                    self.__contextLength,
                )

            return prediction
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def rafInference(
            self,
            sample : pd.core.frame.DataFrame,
            dataset : str,
            softMax : bool = False,
            cosine : bool = True,
            ragPredOnly : bool = False,
            plot : bool = False,
        ) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k)
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            sampleNp : np.ndarray = sample["value"].to_numpy()
            queriedMean, queriedStd = np.mean(queried), np.std(queried)
            sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

            queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
            sampleNormed : np.ndarray = (sampleNp - sampleMean) / (sampleStd + 1e-8)

            difference : float = sampleNormed[0] - queryNormed[-1]
            queryNormed += difference

            newSample : list = queryNormed.tolist() + sampleNormed.tolist()
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": newSample,
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            predictionNormed : np.ndarray = next(iter(self.__predictorRag.predict(sampleGluonts))).quantile(0.5)
            prediction : np.ndarray = (predictionNormed * (sampleStd + 1e-8)) + sampleMean

            if plot:
                Utils.plot(
                    [query.tolist() for query in queriedVectors[0]],
                    "rag.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [queried.tolist()],
                    "ragAvg.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [newSample + predictionNormed.tolist()],
                    "inputAndRag.png",
                    ":",
                    self.__contextLength,
                    rag=True,
                )
                Utils.plot(
                    [sample["value"].tolist() + prediction.tolist()],
                    "pred.png",
                    "-",
                    self.__contextLength,
                )
            return prediction
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def ragOnlyInference(self, sample : pd.core.frame.DataFrame, dataset : str, softMax : bool = False, cosine : bool = True) -> SampleForecast:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries using rag only
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]
        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k)
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            return queried[self.__contextLength:]
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def plotSample(self, sample : pd.core.frame.DataFrame, groundTruth : pd.core.frame.DataFrame, dataset : str):
        """
        Method to plot sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2 or len(groundTruth.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        groundTruth.columns = ["datetime", "value"]

        sampleDict : dict = {
            "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
            "target": sample["value"].tolist(),
        }
        groundTruthDict : dict = {
            "start": TimeManager.convertTimeFormat(groundTruth["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
            "target": groundTruth["value"].tolist()
        }
        sampleGluonts : ListDataset = ListDataset(
            [sampleDict],
            freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat),
        )

        prediction : SampleForecast = next(iter(self.__predictor.predict(sampleGluonts)))

        plot_single(
            sampleDict,
            groundTruthDict,
            prediction,
            context_length=self.__contextLength,
            name="pred",
            show_label=True,
        )
        plt.show()
