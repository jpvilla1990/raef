from pydantic_settings import BaseSettings

class TrainingConfig(BaseSettings):
    """Configuration for training."""
    maxSamplesPerSubdataset: int = 500000
    dataset: str = "lotsaData"
    k: int = 16
    batchSize: int = 3000
    lr: float = 1e-3
    weightDecay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6
    maxIterationsPerEpoch: int = 50
    epochs: int = 3
    checkPoint: str = ""
    lengthCombinations: list = [
        {"contextLength": 32, "predictionLength": 16},
        {"contextLength": 64, "predictionLength": 16},
        {"contextLength": 128, "predictionLength": 16},
    ]

class FineTuningConfig(TrainingConfig):
    """Configuration for fine-tuning."""
    dataset: str = "traffic"
    batchSize: int = 1024
    weightDecay: float = 1e-2
    maxIterationsPerEpoch: int = 500
    epochs: int = 2

class VectorDatabaseConfig(BaseSettings):
    """Configuration for vector database."""
    maxNumberSamples: int = 10000
    scoreThreshold: float = 0.0
    k: int = 16
    collections: dict = {}

class Config(BaseSettings):
    """Configuration for the application."""

    seed: int = 42
    thresholdDistance: float = 1.0
    trainPartition: float = 0.8
    maxTestSamples: int = 10000
    maxNan: float = 0.01
    minUniqueElementRatio: float = 0.02
    training: TrainingConfig = TrainingConfig()
    fineTuning: FineTuningConfig = FineTuningConfig()
    models: dict = {}
    datasets: dict = {}
    monash: dict = {}
    vectorDatabase: VectorDatabaseConfig = VectorDatabaseConfig()
    experiments: dict = {}
    paths: dict = {
        "data": ["data"],
        "datasets": ["data", "datasets"],
        "vectorDatabase": ["data", "vectorDatabase"],
        "RagCAModels": ["data", "models", "RagCA"],
        "lmdbDatabasePath": ["data", "datasets", "lmdbDatabases"],
        "pretrainedModels": ["pretrained"],
    }
    files: dict = {
        "datasets": ["data", "datasets", "datasets.yaml"],
        "databaseTracking": ["data", "databaseTracking.yaml"],
        "results": ["results.yaml"],
        "paramsFineTunedModel": ["data", "models", "RagCA", "MoiraiMoE-finetune.ckpt"],
        "lmdbDatabasesConfig": ["data", "datasets", "lmdbDatabases", "lmdbDatabasesConfig.yaml"],
    }