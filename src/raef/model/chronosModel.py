import uuid
import numpy as np
import torch
from raef.utils.fileSystem import FileSystem
from chronos import BaseChronosPipeline
from raef.vectorDB.vectorDB import vectorDB
from raef.model.raef import RAEF

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
        self.raef : RAEF = RAEF()

    def setInputSpaceCollection(self, collectionName : str, dataset : str):
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

    def predictRaef(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16, extended : bool = False) -> np.ndarray:
        query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
        queried, score = self.queryVector(query, k=self._getConfig()["vectorDatabase"]["k"])
        if queried is not None:
            xContext : torch.Tensor = query.unsqueeze(0)
            queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
            scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)

            augmentedSample, mean, std = self.raef.inference(
                xContext,
                queriedTorch,
                scoreTensor,
                extended,
                thresholdDistance=self._getConfig()["thresholdDistance"],
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
        query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
        queried, score = self.queryVector(query, k=1)
        if queried is not None:
            xContext : torch.Tensor = query
            queriedTorch : torch.Tensor = torch.Tensor(queried)

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