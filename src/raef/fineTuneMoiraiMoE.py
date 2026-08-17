from raef.vectorDB.vectorDBingestion import VectorDBIngestion
from raef.trainingModule.finetuning import Training

training :Training = Training()
training.train("moiraiMoE")