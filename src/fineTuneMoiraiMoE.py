from vectorDB.vectorDBingestion import VectorDBIngestion
from trainingModule.finetuning import Training

training :Training = Training()
training.train("moiraiMoE")