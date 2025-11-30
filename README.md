# Retrieval-Augmented Extended Forecasting


## License

This project is licensed under the Apache License 2.0.  
You can read it here:  
https://www.apache.org/licenses/LICENSE-2.0

## Environment

All tests and experiments were performed in `Ubuntu 24.04.2 LTS`, using an GPU `NVIDIA GeForce RTX 5090`.

## Execution

Execute everything in the root folder using `./python`, this script overwrites the needed configuration to run uv

## Experiments

```bash
./python runExperiment.py -e main -i # Run main experiments with horizon 16 and include Chronos
./python runExperiment.py -e extended -i # Run Extended experiments with horizon 32
./python runExperiment.py -e embedding -i # Run Experiment, comparing embedding space vs input space
./python runExperiment.py -e performance # Run performance experiment to compare inference time
```
If the RAG databases were previously ingested, then the parameter -i can be removed to save time.
The seeds can be changed from the file `src/config.yaml`.

Results are generated in the folder `experiments/`

## Usage

### RAG database Creation
```python
import torch
import numpy as np
from vectorDB.vectorDB import vectorDB

# Ingest samples to RAG database

collectionName = "collectionName"
dataset = "dataset"
ragDatabase = vectorDB = vectorDB()

# In case the collection and dataset need to be delete use the following calls
ragDatabase.deleteDataset(dataset) # Delete samples of corresponding dataset
ragDatabase.deleteCollection(collectionName, dataset) # Delete collection entirely

# Set collection
ragDatabase.setCollection(
    collectionName,
    dataset,
    lambda x : torch.tensor(x).reshape(1, len(x)),
    {"hnsw:space" : "l2"},
)


# Iterate sample by sample and ingest to the RAG database
count = 0
while count < 100:
    timeseries = np.random.rand(144) # Creates a random 144 points vector
    sample = timeseries[:128]
    prediction = timeseries[128:]

    ragDatabase.ingestTimeseries(sample, prediction, dataset)

    count += 1
```

### RAEF (Extended Augmentation)

```python
import numpy as np
import torch
from vectorDB.vectorDB import vectorDB
from model.raef import RAEF

# Load RAG database

collectionName = "collectionName"
dataset = "dataset"
ragDatabase = vectorDB = vectorDB()

# Set collection
ragDatabase.setCollection(
    collectionName,
    dataset,
    lambda x : torch.tensor(x).reshape(1, len(x)),
    {"hnsw:space" : "l2"},
)

raef : RAEF = RAEF()

timeseries = np.random.rand(144) # Creates a random 144 points vector
sample = timeseries[:128]
prediction = timeseries[128:]
k = 16

query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
queried, score = ragDatabase.queryTimeseries(query, k)
if queried is not None:
    xContext : torch.Tensor = query.unsqueeze(0)
    queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
    scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)

    augmentedSample, mean, std = raef.inference(
        xContext,
        queriedTorch,
        scoreTensor,
        thresholdDistance = 1.0, # By default is set to 1.0, but can be modified
    )

    # Augmented sample is normalized, hence mean and std are needed to denormalize the prediction

    prediction = model(augmentedSample)

    stdNp : np.ndarray = std.to("cpu").squeeze(-1).squeeze(-1).numpy()
    meanNp : np.ndarray = mean.to("cpu").squeeze(-1).squeeze(-1).numpy()
    prediction = (prediction * stdNp) + meanNp
else:
    prediction = model(query)
```