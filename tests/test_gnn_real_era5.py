"""Checks for the real-ERA5 GNN stage (``weights/gnn/09_train_real_era5.py``).

These run offline: they exercise the split guard, the residual architecture's
zero-initialised increment head, the architecture selector and the metric
arithmetic. They deliberately do **not** need the gitignored ERA5 archives, so a
clone can run them without a Copernicus account.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GNN_DIR = ROOT / "weights" / "gnn"

SPEC = importlib.util.spec_from_file_location(
    "gnn_real_era5_stage", GNN_DIR / "09_train_real_era5.py"
)
assert SPEC is not None and SPEC.loader is not None
stage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage
SPEC.loader.exec_module(stage)

MODEL_SPEC = importlib.util.spec_from_file_location(
    "gnn_stage5_model", GNN_DIR / "05_st_gnn_model.py"
)
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
model_module = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = model_module
MODEL_SPEC.loader.exec_module(model_module)


@pytest.fixture(scope="module")
def edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


@pytest.fixture(scope="module")
def edge_weight() -> torch.Tensor:
    return torch.ones(4, dtype=torch.float32)


def test_residual_untrained_model_reproduces_persistence(
    edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> None:
    """Epoch 0 must equal the persistence baseline exactly, by construction.

    The increment head is zero-initialised, so the starting point is checkable
    rather than assumed: any later gain is attributable to training, not to an
    accidentally helpful initialisation.
    """
    torch.manual_seed(0)
    model = model_module.ResidualSTGNN(
        n_features=5, hidden_dim=8, embed_dim=4, n_processor_steps=2
    )
    sequence = torch.randn(6, 3, 5)

    with torch.no_grad():
        prediction, embedding = model(sequence, edge_index, edge_weight)

    assert prediction.shape == (3, 5)
    assert embedding.shape == (3, 4)
    torch.testing.assert_close(prediction, sequence[-1])


def test_plain_and_residual_architectures_differ_by_the_skip_connection(
    edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> None:
    """With an identical state dict, only the residual adds the last observation."""
    torch.manual_seed(1)
    residual = model_module.ResidualSTGNN(
        n_features=5, hidden_dim=8, embed_dim=4, n_processor_steps=2, dropout=0.0
    )
    plain = model_module.SpatioTemporalGNN(
        n_features=5, hidden_dim=8, embed_dim=4, n_processor_steps=2
    )
    # Break the zero-init so the decoder emits a non-trivial increment.
    with torch.no_grad():
        residual.decoder_next_step.weight.normal_(0, 0.1)
    plain.load_state_dict(
        {key: value for key, value in residual.state_dict().items()}, strict=True
    )
    sequence = torch.randn(6, 3, 5)

    residual.eval()
    plain.eval()
    with torch.no_grad():
        residual_out, _ = residual(sequence, edge_index, edge_weight)
        plain_out, _ = plain(sequence, edge_index, edge_weight)

    torch.testing.assert_close(residual_out - plain_out, sequence[-1])


def test_build_model_selects_the_requested_architecture() -> None:
    build = stage.build_model

    residual = build("residual", 8, 4, 2, 0.0)
    plain = build("plain", 8, 4, 2, 0.0)

    # Compared by name, not isinstance: the stage module is loaded through
    # importlib in its own module namespace, so its class objects are distinct
    # from this test module's even though they are the same source file.
    assert type(residual).__name__ == "ResidualSTGNN"
    assert type(plain).__name__ == "SpatioTemporalGNN"
    # A residual model must carry the skip connection the plain one does not.
    assert isinstance(residual.dropout, torch.nn.Identity)
    assert not hasattr(plain, "dropout")
    # The residual class must be a genuine extension, not a re-implementation.
    assert issubclass(model_module.ResidualSTGNN, model_module.SpatioTemporalGNN)
    with pytest.raises(ValueError, match="Unknown architecture"):
        build("diffusion", 8, 4, 2, 0.0)


def _hourly(days: tuple[int, ...]) -> list[str]:
    return [
        f"2020-05-{day:02d}T{hour:02d}:00:00Z" for day in days for hour in range(24)
    ]


def test_split_keeps_the_pilot_week_out_of_training() -> None:
    """A contiguous archive must split so every test hour is after every train hour."""
    timestamps = _hourly((15, 16, 17))
    tensor = np.arange(len(timestamps) * 2 * 5, dtype=np.float32).reshape(-1, 2, 5)

    train_tensor, train_times = stage.split_training_window(tensor, timestamps)

    assert train_times[-1] == stage.TRAIN_LAST_HOUR
    assert train_tensor.shape[0] == len(train_times) == 24
    # No training hour may fall on or after the first held-out hour.
    assert all(value < stage.TEST_FIRST_HOUR for value in train_times)
    assert timestamps[len(train_times)] == stage.TEST_FIRST_HOUR


def test_split_refuses_a_gap_between_training_and_the_pilot() -> None:
    """A missing day must fail loudly rather than silently train on the pilot week."""
    timestamps = _hourly((14, 16, 17))
    tensor = np.arange(len(timestamps) * 2 * 5, dtype=np.float32).reshape(-1, 2, 5)

    with pytest.raises(ValueError, match="expected 2020-05-15T23:00:00Z"):
        stage.split_training_window(tensor, timestamps)


def test_split_refuses_a_window_that_never_reaches_the_pilot() -> None:
    timestamps = [f"2020-04-01T{hour:02d}:00:00Z" for hour in range(24)]

    with pytest.raises(ValueError, match="do not cover the pilot start"):
        stage.split_training_window(np.zeros((24, 2, 5), dtype=np.float32), timestamps)


def test_evaluate_reports_physical_errors_and_improvements() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(0.0, 1.0, size=(4, 3, 5)).astype(np.float64)
    # Persistence is the previous hour, so it has a real, non-zero error.
    shifted = np.roll(actual, 1, axis=0)
    persistence = shifted.copy()
    smoothing = shifted + 0.5
    gnn = shifted.copy()
    # Make the GNN worse than persistence on temperature, and equal to it elsewhere.
    gnn[..., 0] = actual[..., 0] + 2.0
    gnn[..., 1:] = persistence[..., 1:]

    frame = stage.evaluate(gnn, persistence, smoothing, actual)

    assert list(frame["feature"]) == stage.FEATURE_NAMES
    assert all(row == 4 * 3 for row in frame["prediction_count"])
    temperature = frame.iloc[0]
    assert temperature["gnn_rmse"] > temperature["persistence_rmse"]
    assert temperature["gnn_vs_persistence_improvement_pct"] < 0
    # The four features the GNN copies from persistence must tie exactly at 0 %.
    for name in ("pressure", "humidity", "wind_speed", "precipitation"):
        row = frame[frame["feature"] == name].iloc[0]
        assert row["gnn_rmse"] == pytest.approx(row["persistence_rmse"])
        assert row["gnn_vs_persistence_improvement_pct"] == pytest.approx(0.0)


def test_improvement_is_defined_when_a_baseline_is_already_perfect() -> None:
    """A perfect baseline is degenerate; the table must not divide by zero."""
    assert stage.improvement_pct(0.0, 1.0) == 0.0
    assert stage.improvement_pct(0.0, 0.0) == 0.0
    assert stage.improvement_pct(2.0, 1.0) == pytest.approx(50.0)
    assert stage.improvement_pct(1.0, 2.0) == pytest.approx(-100.0)


def test_persistence_loss_weighting_equalises_every_feature() -> None:
    """1/persistence_MSE weighting must make each feature contribute equally.

    The point of the weighting is that no single feature can dominate the loss and
    let the model give away the others. After weighting, every feature's own
    persistence error contributes the same amount, so the per-feature shares must
    be uniform.
    """
    rng = np.random.default_rng(1)
    tensor = rng.normal(0.0, 1.0, size=(50, 4, 5))
    increments = tensor[1:] - tensor[:-1]
    per_feature = (increments**2).mean(axis=(0, 1))
    weights = (1.0 / per_feature) / (1.0 / per_feature).mean()

    contributions = per_feature * weights
    share = contributions / contributions.sum()

    assert share == pytest.approx(np.full(5, 1.0 / 5), rel=1e-9)
    # Unweighted, one feature would swamp the rest; check the weights are not flat.
    assert not np.allclose(weights, 1.0)
    # The weighted persistence score is the harmonic mean of the raw persistence
    # MSEs -- strictly below the arithmetic mean, which is the point: the least
    # predictable feature stops setting the scale of the objective.
    assert float(((increments**2) * weights).mean()) == pytest.approx(
        len(per_feature) / np.sum(1.0 / per_feature)
    )
