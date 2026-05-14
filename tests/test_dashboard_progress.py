import pytest

from dashboard_server import infer_phase, infer_progress_floor


@pytest.mark.parametrize(
    ("line", "expected_phase", "expected_floor"),
    [
        (
            "Training state statistics saved to data/training_state.json",
            "training_setup",
            None,
        ),
        (
            "--- Training Seed 3 (3/5) ---",
            "training",
            46,
        ),
        (
            "Best model globally (val_loss: 0.5836) saved to data/checkpoints/best_mvp_model.pt",
            "finalizing",
            97,
        ),
    ],
)
def test_progress_log_line_inference(line, expected_phase, expected_floor):
    assert infer_phase(line) == expected_phase
    assert infer_progress_floor(line) == expected_floor


@pytest.mark.parametrize(
    "line",
    [
        "Training state statistics saved to data/training_state.json",
        "Applying data mirroring to training set...",
        "Scaler saved to data/checkpoints/scaler.pkl",
    ],
)
def test_training_setup_lines_do_not_jump_to_finalizing(line):
    assert infer_phase(line) != "finalizing"


def test_pipeline_completion_reaches_complete_floor():
    assert infer_progress_floor("PIPELINE COMPLETED SUCCESSFULLY") == 100
