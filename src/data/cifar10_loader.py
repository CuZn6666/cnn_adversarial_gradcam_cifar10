from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from configs.default_config import (
    CIFAR10_ARCHIVE_NAME,
    CIFAR10_EXTRACTED_DIR,
    CIFAR10_MD5,
    CIFAR10_URL,
    DATA_DIR,
)


def compute_md5(file_path: Path) -> str:
    """Compute the MD5 checksum of a file."""
    md5 = hashlib.md5()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


def download_and_extract_cifar10(data_dir: Path = DATA_DIR) -> Path:
    """
    Download and extract the official CIFAR-10 Python archive if necessary.

    Returns:
        Path to the extracted CIFAR-10 batch directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / CIFAR10_ARCHIVE_NAME
    extracted_dir = data_dir / CIFAR10_EXTRACTED_DIR

    if extracted_dir.exists():
        return extracted_dir

    if not archive_path.exists():
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print(f"Downloaded archive to: {archive_path}")

    archive_md5 = compute_md5(archive_path)
    if archive_md5 != CIFAR10_MD5:
        raise ValueError(
            "CIFAR-10 archive checksum mismatch. "
            f"Expected {CIFAR10_MD5}, but obtained {archive_md5}."
        )

    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")

    if not extracted_dir.exists():
        raise FileNotFoundError("CIFAR-10 extraction failed.")

    print(f"Extracted dataset to: {extracted_dir}")
    return extracted_dir


def unpickle_batch(file_path: Path) -> dict:
    """Read one CIFAR-10 pickled batch file."""
    with file_path.open("rb") as file:
        return pickle.load(file, encoding="bytes")


def load_batch(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one CIFAR-10 batch.

    Returns:
        images: float32 array with shape (N, 3, 32, 32), values in [0, 1]
        labels: int64 array with shape (N,)
    """
    batch = unpickle_batch(file_path)

    images = np.asarray(batch[b"data"], dtype=np.float32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)

    images = images.reshape(-1, 3, 32, 32) / 255.0

    return images, labels


def load_label_names(dataset_dir: Path) -> list[str]:
    """Load human-readable CIFAR-10 class names."""
    metadata = unpickle_batch(dataset_dir / "batches.meta")
    return [name.decode("utf-8") for name in metadata[b"label_names"]]


def load_cifar10(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load the complete CIFAR-10 dataset.

    Returns:
        x_train: shape (50000, 3, 32, 32), float32, values in [0, 1]
        y_train: shape (50000,), int64
        x_test: shape (10000, 3, 32, 32), float32, values in [0, 1]
        y_test: shape (10000,), int64
        class_names: list of 10 class names
    """
    dataset_dir = download_and_extract_cifar10(data_dir)

    train_batches = [
        load_batch(dataset_dir / f"data_batch_{index}")
        for index in range(1, 6)
    ]

    x_train = np.concatenate([batch[0] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[1] for batch in train_batches], axis=0)

    x_test, y_test = load_batch(dataset_dir / "test_batch")
    class_names = load_label_names(dataset_dir)

    return x_train, y_train, x_test, y_test, class_names
