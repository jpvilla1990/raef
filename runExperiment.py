import click
from utils.fileSystem import FileSystem
from utils.utils import Utils
from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

fileSystem : FileSystem = FileSystem()
vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

config : dict = fileSystem._getConfig()
resultsFile : str = fileSystem._getFiles()["results"]

def recordResults(key: str, value: dict):
    results : dict = Utils.readYaml(resultsFile)
    if type(results) != dict:
        results = dict()
    results[key] = value
    Utils.writeYaml(
        resultsFile,
        results,
    )

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
                    report : dict = dict()
                    if model == "MoiraiMoE":
                        report = evaluation.evaluateMoiraiMoERagCA(
                            contextLength=context,
                            predictionLength=horizon,
                            numberSamples=100,
                            dataset=dataset,
                            collection=ragDatabaseFullName,
                            fineTunedModel=f"MoiraiMoE-finetune-{dataset}.ckpt",
                            inputSpace=inputSpace,
                        )
                    elif model == "ChronosT5":
                        report = evaluation.evaluateChronosRagLeveling(
                            contextLength=context,
                            predictionLength=horizon,
                            dataset=dataset,
                            collection=ragDatabaseFullName,
                            bolt=False,
                        )
                    elif model == "ChronosBolt":
                        report = evaluation.evaluateChronosRagLeveling(
                            contextLength=context,
                            predictionLength=horizon,
                            dataset=dataset,
                            collection=ragDatabaseFullName,
                            bolt=True,
                        )

                    recordResults(
                        f"{experiment}_{context}_{horizon}_{model}_{ragDatabase}_{dataset}",
                        {
                            "experiment": experiment,
                            "context_length": context,
                            "prediction_length": horizon,
                            "model": model,
                            "rag_database": ragDatabaseFullName,
                            "dataset": dataset,
                            "similarity_metric": scenario["ragDatabases"][ragDatabase]["inputSpace"],
                            "report": report,
                        },
                    )

if __name__ == "__main__":
    main()