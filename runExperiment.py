import click
from utils.fileSystem import FileSystem
from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

fileSystem : FileSystem = FileSystem()
vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

config : dict = fileSystem._getConfig()

@click.command()
@click.option(
    "--experiment",
    "-e",
    type=click.Choice(list(config["experiments"].keys()), case_sensitive=False),
    required=True,
    help="Experiment to run."
)
@click.option(
    "--ingest-rag",
    "-i",
    is_flag=True,
    help="Ingest RAG databases.",
    default=False,
)
def main(experiment, ingest_rag):
    scenario : dict = config["experiments"][experiment]
    for context_horizon in scenario["context_horizons"]:
        context = context_horizon[0]
        horizon = context_horizon[1]

        for ragDatabase  in list(scenario["ragDatabases"].keys()):
            ragDatabaseFullName = f"{ragDatabase}_{context}_{horizon}"
            inputSpace = scenario["ragDatabases"][ragDatabase]["inputSpace"]
            if ingest_rag:
                vectorDBingestion.ingestDatasetsMoiraiMoE(ragDatabaseFullName, inputSpace)
            for model in scenario["models"]:
                for dataset in scenario["datasets"]:
                    evaluation.evaluateMoiraiMoERagCA(
                        contextLength=context,
                        predictionLength=horizon,
                        numberSamples=100,
                        dataset=dataset,
                        collection=ragDatabaseFullName,
                        fineTunedModel=f"MoiraiMoE-finetune-{dataset}.ckpt",
                        inputSpace=inputSpace,
                    )
                    print("experiment")
                    print(context)
                    print(horizon)
                    print(model)
                    print(ragDatabase)
                    print(dataset)
                    print(ragDatabaseFullName)
                    print(scenario["ragDatabases"][ragDatabase]["inputSpace"])
                    print("-------------------")

if __name__ == "__main__":
    main()