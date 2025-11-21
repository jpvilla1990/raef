import os
import concurrent.futures
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from utils.fileSystem import FileSystem
from utils.utils import Utils
from model.moiraiMoe import MoiraiMoE
from datasetsModule.datasetTrainingIterator import DatasetTrainingIterator
from utils.utils import Utils

class TrainingMoiraiMoE(L.LightningModule):
    def __init__(
        self,
        dataset : str,
        lr : float = 1e-3,
        numberSamples : int = 100,
        batchSize : int = 1,
        weightDecay : float = 1e-1,
        betas : tuple = (0.9, 0.98),
        eps : float = 1e-6,
    ):
        super().__init__()
        self.__dataset : str = dataset
        self.__lr : float = lr
        self.numberSamples : int = numberSamples
        self.__batchSize : int = batchSize
        self.__weightDecay : float = weightDecay
        self.__betas : tuple = betas
        self.__eps : float = eps

        self.backBoneModel : MoiraiMoE = MoiraiMoE(
            numSamples = self.numberSamples,
            batchSize = self.__batchSize,
            createPredictor=False,
            frozen=False,
        )
        self.model = self.backBoneModel.model

        print(self.backBoneModel.model.module.parameters())

    def __getMASE(self, context : torch.Tensor, groundTruth : torch.Tensor, prediction : torch.Tensor) -> torch.Tensor:
        """
        Method to calculate MEAN ABSOLUTE SCALED ERROR
        """
        meanAbsoluteError : torch.Tensor = (torch.abs(prediction - groundTruth)).mean(dim=1)

        meanAbsoluteDeviation : torch.Tensor = torch.abs(context[:, :-1] - context[:, 1:]).mean(dim=1)

        mase : torch.Tensor = meanAbsoluteError / meanAbsoluteDeviation

        mask = torch.isfinite(mase)

        return mase[mask].mean()

    def training_step(
            self,
            batch : tuple[torch.Tensor, int],
            batch_idx : int,
        ) -> torch.Tensor:
        """
        Training step for the model.
        :param batch: Batch of data to be used for training. [B, L]
        :return: Loss value for the batch.
        """
        print(f"Batch: {batch_idx}")
        batch, contextLength = batch
        if contextLength.shape == torch.Size([1]):
            contextLength = contextLength.item()
        else:
            contextLength = contextLength[-1].item()
        batch = batch.squeeze(0)
        predictionLength : int = batch.shape[1] - contextLength

        xContext : torch.Tensor = batch[:, :contextLength]
        xTarget : torch.Tensor = batch[:, contextLength:]

        pred : torch.Tensor = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 : concurrent.futures._base.Future = executor.submit(
                self.backBoneModel.forwardRagCA,
                xContext,
                True,
            )

            pred = future1.result()

        logProbPred : torch.Tensor = -pred.log_prob(xTarget.unsqueeze(1))[:,-2,:]
        predSample : torch.Tensor = pred.sample(torch.Size((self.numberSamples,))).mean(dim=0)[:,-2,:].clone()

        masePred : torch.Tensor = self.__getMASE(xContext, xTarget, predSample)
        loss : torch.Tensor = logProbPred
        loss = loss.mean()
        print(loss)
        print(f"MASE Pred: {masePred}")

        Utils.plot(
            [
                xContext[0].tolist() + xContext[0].tolist() + xTarget[0].tolist(),
            ],
            "train_pred.png",
            "-",
            contextLength,
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.__lr,
            weight_decay=self.__weightDecay,
            betas=self.__betas,
            eps=self.__eps,
        )

class Training(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()
        self.__batchSize : int = self._getConfig()["finetuning"]["batchSize"]
        self.__epochs : int = self._getConfig()["finetuning"]["epochs"]

    def __loadReport(self, report : str) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()[report]
        )
        return reports if type(reports) == dict else dict()

    def __writeReport(self, enztry : dict, report : str):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()[report],
            self.__loadReport(report) | entry,
        )

    def saveModelState(self, parametersFile : str):
        """
        Method to get MoiraMoE state
        """
        parametersFile = os.path.join(self._getPaths()["RagCAModels"], parametersFile)
        model = TrainingMoiraiMoE.load_from_checkpoint(
            parametersFile,
            dataset=self._getConfig()["finetuning"]["dataset"],
            batchSize=self.__batchSize,
            lr=float(self._getConfig()["finetuning"]["lr"]),
            weightDecay=float(self._getConfig()["finetuning"]["weightDecay"]),
            betas=(float(self._getConfig()["finetuning"]["beta1"]), float(self._getConfig()["finetuning"]["beta2"])),
            eps=float(self._getConfig()["finetuning"]["eps"]),
        ).model
        torch.save(model.state_dict(), self._getFiles()["paramsFineTunedModel"])

    def train(
            self,
            modelName : str,
        ):
        """
        Method to train the RagCA model
        """
        model : TrainingMoiraiMoE = TrainingMoiraiMoE(
            dataset=self._getConfig()["finetuning"]["dataset"],
            batchSize=self.__batchSize,
            lr=float(self._getConfig()["finetuning"]["lr"]),
            weightDecay=float(self._getConfig()["finetuning"]["weightDecay"]),
            betas=(float(self._getConfig()["finetuning"]["beta1"]), float(self._getConfig()["finetuning"]["beta2"])),
            eps=float(self._getConfig()["finetuning"]["eps"]),
        )
        tsDataset = DatasetTrainingIterator(
            self._getConfig(),
            Utils.readYaml(self._getFiles()["datasets"]),
            balanced=True,
        )
        tsLoader = DataLoader(tsDataset, batch_size=1, num_workers=0)
        trainer = L.Trainer(
            max_epochs=self.__epochs,
            accelerator="auto",
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self._getPaths()["RagCAModels"],
                    filename=f"MoiraiMoE-finetune-{self._getConfig()["finetuning"]["dataset"]}-{{epoch:02d}}-{{step:06d}}",
                    save_top_k=-1,
                    every_n_train_steps=1,
                    save_on_train_epoch_end=False,
                ),
            ],
        )
        if self._getConfig()["finetuning"]["checkpoint"] != "":
            trainer.fit(
                model,
                tsLoader, 
                ckpt_path = os.path.join(
                    self._getPaths()["RagCAModels"],
                    self._getConfig()["finetuning"]["checkpoint"],
                ),
            )
        else:
            trainer.fit(
                model,
                tsLoader, 
            )

    def test(self):
        tsDataset = DatasetTrainingIterator(
            self._getConfig(),
            Utils.readYaml(self._getFiles()["datasets"]),
            balanced=True,
        )
        a = tsDataset.test(0)

if __name__ == "__main__":
    training : Training = Training()
    training.train()