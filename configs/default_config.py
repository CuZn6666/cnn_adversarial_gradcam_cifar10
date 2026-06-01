from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"  ##Data Set Repository
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" ## Default Save Location for Images

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