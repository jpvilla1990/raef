import uuid
import numpy as np
import torch
from typing import List
from utils.fileSystem import FileSystem
from chronos import BaseChronosPipeline
from vectorDB.vectorDB import vectorDB
from model.ragCrossAttention import RagCrossAttention
from utils.utils import Utils

class Chronos(FileSystem):
    """
    Datasets used pretrained:
    electricityUCI
    m4
    nn5
    fredmd
    """
    def __init__(
        self,
        bolt : bool = False,
        frozen : bool = False,
        loadPretrainedModel : bool = True,
    ):
        self.__bolt : bool = bolt
        self.__modelName : str = "amazon/chronos-bolt-small" if bolt else "amazon/chronos-t5-small"
        super().__init__()
        self.__model : BaseChronosPipeline = BaseChronosPipeline.from_pretrained(
            self.__modelName,
            device_map = "cuda" if torch.cuda.is_available() else "cpu",
        )
        if frozen:
            for param in self.__model.inner_model.parameters():
                param.requires_grad = False
            for param in self.__model.model.parameters():
                param.requires_grad = False
        self.__vectorDB : vectorDB = vectorDB()
        self.__modelRagCA : RagCrossAttention = RagCrossAttention(
            patchSize=16,
            pretrainedModel=self._getFiles()["paramsRagCA"],
            loadPretrainedModel=loadPretrainedModel,
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

    def queryVector(self, sample : np.ndarray, k : int = 1) -> tuple:
        """
        Method to query vector
        """
        return self.__vectorDB.queryTimeseries(sample, k)

    def predict(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16) -> np.ndarray:
        sample = torch.tensor(sample) if type(sample) == np.ndarray else sample
        import time
        start = time.perf_counter()
        prediction = self.__model.predict_quantiles(
            context=sample,
            prediction_length=predictionLength,
        )
        end = time.perf_counter()

        #print(f"Inference time: {(end - start) * 1000:.3f} ms")
        return prediction[1].squeeze().numpy()

    def predictTraining(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16) -> np.ndarray:
        """
        Method to predict with training
        """
        sample = torch.tensor(sample) if type(sample) == np.ndarray else sample
        return self.__model.predict_quantiles(
            context=sample,
            prediction_length=predictionLength,
        )[1].squeeze()

    def forwardLoss(self, context : torch.Tensor, mean : torch.Tensor, std : torch.Tensor,  target : torch.Tensor) -> np.ndarray:
        if self.__bolt:
            return self.forwardLossBolt(context, mean, std, target)
        else:
            return self.forwardLossT2(context, mean, std, target)

    def forwardLossBolt(self, context : torch.Tensor, mean : torch.Tensor, std : torch.Tensor, target : torch.Tensor) -> np.ndarray:
        """
        Method to forward the model
        """
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.__model.inner_model.encode(
            context=context, mask=None
        )
        sequence_output = self.__model.inner_model.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.__model.inner_model.num_quantiles,
            self.__model.inner_model.chronos_config.prediction_length,
        )
        quantile_preds = self.__model.inner_model.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        # normalize target
        target, _ = self.__model.inner_model.instance_norm(target, loc_scale)
        target = target.unsqueeze(1)  # type: ignore
        assert self.__model.inner_model.chronos_config.prediction_length >= target.shape[-1]

        target = target.to(quantile_preds.device)
        target_mask = None
        target_mask = (
            target_mask.unsqueeze(1).to(quantile_preds.device)
            if target_mask is not None
            else ~torch.isnan(target)
        )
        target[~target_mask] = 0.0

        # pad target and target_mask if they are shorter than model's prediction_length
        if self.__model.inner_model.chronos_config.prediction_length > target.shape[-1]:
            padding_shape = (
                *target.shape[:-1],
                self.__model.inner_model.chronos_config.prediction_length - target.shape[-1],
            )
            target = torch.cat(
                [target, torch.zeros(padding_shape).to(target)], dim=-1
            )
            target_mask = torch.cat(
                [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
            )

        loss = (
            2
            * torch.abs(
                (target - quantile_preds)
                * (
                    (target <= quantile_preds).float()
                    - self.__model.inner_model.quantiles.view(1, self.__model.inner_model.num_quantiles, 1)  # type: ignore
                )
            )
            * target_mask.float()
        )
        loss = loss.mean(dim=-2)  # Mean over prediction horizon
        loss = loss.sum(dim=-1)  # Sum over quantile levels
        loss = loss.mean()  # Mean over batch
        return loss

    def forwardLossT2(self, context : torch.Tensor, mean : torch.Tensor, std : torch.Tensor, target : torch.Tensor) -> np.ndarray:
        """
        Method to forward the model
        """
        context = context.to("cpu")
        quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device=context.device)
        prediction_samples = (
            self.__model.predict(context, prediction_length=target.shape[-1])
            .swapaxes(1, 2)
        )
        quantiles_pred = torch.quantile(
            prediction_samples,
            q=torch.tensor(quantiles, dtype=prediction_samples.dtype),
            dim=-1,
        ).permute(1, 2, 0)

        target = target.to(quantiles_pred.device)
        target = (target - mean.squeeze(-1).squeeze(-1).to(quantiles_pred.device)) / std.squeeze(-1).squeeze(-1).to(quantiles_pred.device)
        quantiles = quantiles.view(1, 1, -1)  # Shape: [1, 9, 1] for broadcasting

        target = target.unsqueeze(-1)  # Remove batch dimension if present

        # Broadcasted quantile loss (pinball loss)
        loss = (
            2
            * torch.abs(
                (target - quantiles_pred)
                * ((target <= quantiles_pred).float() - quantiles)
            )
        )
        loss = loss.mean(dim=-2)  # Mean over prediction horizon
        loss = loss.sum(dim=-1)  # Sum over quantile levels
        loss = loss.mean()  # Mean over batch
        return loss

    def predictPredictionLength(self, sample : np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Method to predict without prediction length
        """
        sample = torch.tensor(sample) if type(sample) == np.ndarray else sample
        return self.__model.predict_quantiles(
            context=sample,
            prediction_length=None,
        )[1].squeeze().numpy()

    def queryBatchVector(self, sample : np.ndarray | torch.Tensor, k) -> tuple:
        """
        Method to query batch vector
        """
        batch : torch.tensor = torch.tensor(sample, dtype=torch.float32)
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

    def predictRag(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16, extended : bool = False) -> np.ndarray:
        contextLength : int = sample.shape[-1]
        query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
        queried, score = self.queryVector(query, k=self._getConfig()["vectorDatabase"]["k"])
        if queried is not None:
            xContext : torch.Tensor = query.unsqueeze(0)
            queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
            scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)

            augmentedSample, mean, std = self.__modelRagCA.inference(
                xContext,
                queriedTorch,
                scoreTensor,
                extended
            )

            id : str = str(uuid.uuid4())

            #Utils.plot(
            #    [
            #        augmentedSample.squeeze().tolist(),
            #    ],
            #    "images/augmentedSample" + id + ".png",
            #    "-",
            #    contextLength + contextLength + predictionLength,
            #)

            return ((self.predict(
                augmentedSample.squeeze().to("cpu"),
                predictionLength,
            ) * std.to("cpu").squeeze(-1).squeeze(-1).numpy()) + mean.to("cpu").squeeze(-1).squeeze(-1).numpy()).squeeze()
        else:
            return self.predict(
                sample,
                predictionLength,
            )

    def predictRaf(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16) -> np.ndarray:
        contextLength : int = sample.shape[-1]
        query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
        queried, score = self.queryVector(query, k=1)
        if queried is not None:
            xContext : torch.Tensor = query
            queriedTorch : torch.Tensor = torch.Tensor(queried)
            scoreTensor : torch.Tensor = torch.Tensor(score)

            queriedTorch = queriedTorch.mean(dim=0)

            difference : float = xContext[0] - queriedTorch[-1]
            queriedTorch += difference
            augmentedSample : torch.Tensor = torch.cat(
                (queriedTorch, xContext),
                dim=0,
            )

            return self.predict(
                augmentedSample,
                predictionLength,
            )
        else:
            return self.predict(
                sample,
                predictionLength,
            )