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
@click.option(
    "--force-experiments",
    "-f",
    is_flag=True,
    help="Force Experiments.",
    default=False,
)
def main(experiment, ingest_rag, force_experiments):
    scenario : dict = config["experiments"][experiment]
    seed : int = config["seed"]
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
                    experimentKey = f"{experiment}_{context}_{horizon}_{model}_{ragDatabase}_{dataset}_seed{seed}"
                    results : dict = Utils.readYaml(resultsFile)
                    if experimentKey in results and not force_experiments:
                        print(f"Skipping experiment {experimentKey} as it already exists in results.")
                        continue
                    print(f"Running experiment {experimentKey}")
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
                        experimentKey,
                        {
                            "experiment": experiment,
                            "context_length": context,
                            "seed": seed,
                            "prediction_length": horizon,
                            "model": model,
                            "rag_database": ragDatabaseFullName,
                            "dataset": dataset,
                            "input_space": scenario["ragDatabases"][ragDatabase]["inputSpace"],
                            "similarity_metric": config["vectorDatabase"]["collections"][ragDatabaseFullName]['metric'],
                            "report": report,
                        },
                    )

if __name__ == "__main__":
    main()