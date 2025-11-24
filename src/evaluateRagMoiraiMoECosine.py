from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

contextLength : int = 32
predictionLength : int = 16

vectorDBingestion.ingestDatasetsMoiraiMoE(f"domainDatasetCosine_{contextLength}_{predictionLength}", False)

loadPretrainedModel : bool = False

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="ET",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-ET-epoch=00-step=000200.ckpt",
    inputSpace=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="huaweiCloud",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-huaweiCloud-epoch=00-step=001000.ckpt",
    inputSpace=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="power",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-power-epoch=00-step=000010.ckpt",
    inputSpace=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="traffic",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-traffic-epoch=00-step=000008.ckpt",
    inputSpace=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="fredMd",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-fredMd-epoch=00-step=000100.ckpt",
    inputSpace=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=predictionLength,
    numberSamples=100,
    dataset="electricityUCI",
    collection=f"domainDatasetL2_{contextLength}_{predictionLength}",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-electricityUCI-epoch=00-step=003500.ckpt",
    inputSpace=False,
)