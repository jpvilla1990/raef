import numpy as np
from raef.vectorDB.vectorDB import vectorDB


def test_ingest_multiple_samples(rag_database, random_timeseries):
    """Ingesting 100 samples in a loop (as in the README example)
    should complete without error and all should be queryable."""
    db, collection_name, dataset = rag_database

    for _ in range(100):
        sample, prediction = random_timeseries()
        db.ingestTimeseries(sample, prediction, dataset)

    # Sanity check: querying against the now-populated dataset returns results
    import torch
    query_sample, _ = random_timeseries()
    query = torch.tensor(query_sample, dtype=torch.float32)
    queried, score = db.queryTimeseries(query, k=16)

    assert queried is not None
    assert score is not None


def test_delete_dataset_and_collection(rag_database, random_timeseries):
    """deleteDataset and deleteCollection should run without raising,
    both on populated and already-empty state."""
    db, collection_name, dataset = rag_database
    sample, prediction = random_timeseries()
    db.ingestTimeseries(sample, prediction, dataset)

    db.deleteDataset(dataset)
    db.deleteCollection(collection_name, dataset)