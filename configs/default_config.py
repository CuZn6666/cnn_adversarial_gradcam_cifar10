from dataclasses import dataclass
import math
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"  ##Data Set Repository
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" ## Default Save Location for Images
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

# Reproducibility
SEED = 42 ## Ensuring the reproducibility of batch order

# Data pipeline
BATCH_SIZE = 64 ## Default batch size for subsequent training
NUM_CLASSES = 10
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

# Data convention used throughout the project
IMAGE_RANGE = (0.0, 1.0)
IMAGE_FORMAT = "NCHW" 
##Specify the model input as NCHW (N = number of samples, C = number of channels, i.e., the three RGB channels, H = image height 32, W = image width 32)

# Official CIFAR-10 Python archive
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a" ## Verify that the downloaded data file is not corrupted


@dataclass(frozen=True)
class BaselineConfig:
    """Small baseline configuration for local WP5 smoke runs."""

    learning_rate: float = 5e-4
    batch_size: int = 8
    epochs: int = 1
    seed: int = SEED
    train_subset_size: int | None = 64
    eval_subset_size: int | None = 32
    checkpoint_dir: Path | str = CHECKPOINT_DIR
    log_dir: Path | str = LOG_DIR
    figure_dir: Path | str = FIGURE_DIR

    def __post_init__(self) -> None:
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number.")

        for field_name in ("batch_size", "epochs"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

        for field_name in ("train_subset_size", "eval_subset_size"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{field_name} must be a positive integer or None."
                )

        for field_name in ("checkpoint_dir", "log_dir", "figure_dir"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))


BASELINE_CONFIG = BaselineConfig()
