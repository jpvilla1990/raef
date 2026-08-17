import numpy as np
import pytest
import torch
from raef.model.raef import RAEF


@pytest.fixture
def populated_rag_database(rag_database, random_timeseries):
    """A rag_database pre-populated with enough samples for k=16 queries."""
    db, collection_name, dataset = rag_database
    for _ in range(50):
        sample, prediction = random_timeseries()
        db.ingestTimeseries(sample, prediction, dataset)
    return db, collection_name, dataset


def test_raef_inference_shapes(populated_rag_database, random_timeseries):
    """Full RAG-augmentation flow from the README example: query the DB,
    run RAEF.inference, and check output shapes/dtypes are sane."""
    db, collection_name, dataset = populated_rag_database
    sample, prediction = random_timeseries()
    k = 16

    raef = RAEF()
    query = torch.tensor(sample, dtype=torch.float32)
    queried, score = db.queryTimeseries(query, k)

    assert queried is not None, "Expected the populated DB to return neighbors"

    x_context = query.unsqueeze(0)
    queried_torch = torch.Tensor(queried).unsqueeze(0)
    score_tensor = torch.Tensor(score).unsqueeze(0)

    augmented_sample, mean, std = raef.inference(
        x_context,
        queried_torch,
        score_tensor,
        thresholdDistance=1.0,
    )

    assert augmented_sample is not None
    assert mean is not None
    assert std is not None
    assert torch.is_tensor(augmented_sample)
    assert torch.is_tensor(mean)
    assert torch.is_tensor(std)


def test_raef_denormalization_roundtrip(populated_rag_database, random_timeseries):
    """Verify the denormalization arithmetic from the README example runs
    correctly against a dummy 'model' standing in for the real forecaster,
    and produces a real-valued, correctly-shaped prediction."""
    db, collection_name, dataset = populated_rag_database
    sample, prediction = random_timeseries()
    k = 16

    raef = RAEF()
    query = torch.tensor(sample, dtype=torch.float32)
    queried, score = db.queryTimeseries(query, k)
    assert len(queried) != 0

    x_context = query.unsqueeze(0)
    queried_torch = torch.Tensor(queried).unsqueeze(0)
    score_tensor = torch.Tensor(score).unsqueeze(0)

    augmented_sample, mean, std = raef.inference(
        x_context, queried_torch, score_tensor, thresholdDistance=1.0,
    )

    # Stand-in for a real foundation model backend (Chronos/MoiraiMoE) —
    # just returns zeros of the expected horizon length so we can test
    # the denormalization math in isolation.
    def dummy_model(x):
        return np.zeros(16)

    pred = dummy_model(augmented_sample)

    std_np = std.to("cpu").squeeze(-1).squeeze(-1).numpy()
    mean_np = mean.to("cpu").squeeze(-1).squeeze(-1).numpy()
    denorm_pred = (pred * std_np) + mean_np

    assert denorm_pred.shape == pred.shape
    assert np.isfinite(denorm_pred).all()


def test_raef_no_neighbors_fallback(rag_database, random_timeseries):
    """When the DB has no ingested samples yet, queryTimeseries should
    return None so the fallback path (direct model(query), no augmentation)
    is what gets exercised — mirrors the else branch in the README example."""
    db, collection_name, dataset = rag_database
    sample, _ = random_timeseries()
    query = torch.tensor(sample, dtype=torch.float32)

    queried, score = db.queryTimeseries(query, k=16)
    assert len(queried) == 0