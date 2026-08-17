import uuid
import numpy as np
import pytest
import torch
from raef.vectorDB.vectorDB import vectorDB


@pytest.fixture
def rag_database():
    """
    Fresh vectorDB collection/dataset per test, cleaned up afterward
    so tests never collide with each other or leave residue behind.
    """
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    dataset = f"test_dataset_{uuid.uuid4().hex[:8]}"

    db = vectorDB()
    db.setCollection(
        collection_name,
        dataset,
        lambda x: torch.tensor(x).reshape(1, len(x)),
        {"hnsw:space": "l2"},
    )

    yield db, collection_name, dataset

    # Teardown — always attempt cleanup even if the test failed
    try:
        db.deleteDataset(dataset)
    except Exception:
        pass
    try:
        db.deleteCollection(collection_name, dataset)
    except Exception:
        pass


@pytest.fixture
def random_timeseries():
    """144-point series split into a 128-sample context and 16-point target,
    matching the shapes used in both documented examples."""
    def _make():
        timeseries = np.random.rand(144)
        return timeseries[:128], timeseries[128:]
    return _make

def test_ingest_single_sample(rag_database, random_timeseries):
    """A single sample should ingest without error."""
    db, collection_name, dataset = rag_database
    sample, prediction = random_timeseries()

    db.ingestTimeseries(sample, prediction, dataset)
    # No exception raised => ingestion succeeded
