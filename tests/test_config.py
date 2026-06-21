from dataclasses import fields
from pathlib import Path

import pytest

from configs.default_config import BASELINE_CONFIG, BaselineConfig


def test_default_baseline_config_contains_required_fields() -> None:
    field_names = {field.name for field in fields(BaselineConfig)}

    assert field_names == {
        "learning_rate",
        "batch_size",
        "epochs",
        "seed",
        "train_subset_size",
        "eval_subset_size",
        "checkpoint_dir",
        "log_dir",
        "figure_dir",
    }


def test_default_baseline_config_is_valid_for_small_local_runs() -> None:
    assert BASELINE_CONFIG.learning_rate > 0
    assert BASELINE_CONFIG.batch_size > 0
    assert BASELINE_CONFIG.epochs == 1
    assert isinstance(BASELINE_CONFIG.seed, int)
    assert BASELINE_CONFIG.train_subset_size == 64
    assert BASELINE_CONFIG.eval_subset_size == 32


def test_baseline_config_normalizes_output_directories_to_paths() -> None:
    config = BaselineConfig(
        checkpoint_dir="custom/checkpoints",
        log_dir=Path("custom/logs"),
        figure_dir="custom/figures",
    )

    assert isinstance(config.checkpoint_dir, Path)
    assert isinstance(config.log_dir, Path)
    assert isinstance(config.figure_dir, Path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_message"),
    [
        ("learning_rate", 0.0, "learning_rate must be"),
        ("learning_rate", float("inf"), "learning_rate must be"),
        ("batch_size", 0, "batch_size must be"),
        ("epochs", -1, "epochs must be"),
        ("seed", 1.5, "seed must be"),
        ("train_subset_size", 0, "train_subset_size must be"),
        ("eval_subset_size", -1, "eval_subset_size must be"),
        ("checkpoint_dir", "", "checkpoint_dir must be"),
        ("log_dir", 123, "log_dir must be"),
        ("figure_dir", None, "figure_dir must be"),
    ],
)
def test_baseline_config_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        BaselineConfig(**{field_name: invalid_value})
