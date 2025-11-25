# Retrieval-Augmented Extended Forecasting


## License

This project is licensed under the Apache License 2.0.  
You can read it here:  
https://www.apache.org/licenses/LICENSE-2.0


## Execution

Execute everything in the root folder using `./python`, this script overwrites the needed configuration to run uv

## Experiments

```bash
./python runExperiment.py -e main -i # Run main experiments with horizon 16 and include Chronos
./python runExperiment.py -e extended -i # Run Extended experiments with horizon 32
./python runExperiment.py -e embedding -i # Run Experiment, comparing embedding space vs input space
```
If the RAG databases were previously ingested, then the parameter -i can be removed to save time.