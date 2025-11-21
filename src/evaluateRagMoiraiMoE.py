from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

contextLength : int = 128

#vectorDBingestion.ingestDatasetsMoiraiMoE(f"moiraiMoECosine_{contextLength}_16")

vectorDBingestion.ingestDatasetsMoiraiMoE(f"moiraiMoEL2_{contextLength}_16")

loadPretrainedModel : bool = False

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-ET-epoch=00-step=000200.ckpt",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-huaweiCloud-epoch=00-step=001000.ckpt",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-power-epoch=00-step=000010.ckpt",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=contextLength,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    loadPretrainedRagCA=loadPretrainedModel,
    fineTunedModel="MoiraiMoE-finetune-traffic-epoch=00-step=000008.ckpt",
)

print("chronos T5")

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="ET",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=False,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="huaweiCloud",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=False,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="power",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=False,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="traffic",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=False,
)

print("chronos Bolt")

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="ET",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=True,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="huaweiCloud",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=True,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="power",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=True,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=contextLength,
    predictionLength=16,
    dataset="traffic",
    collection=f"moiraiMoERafL2_{contextLength}_16",
    bolt=True,
)