import os.path

import pytest
import torch

from mnist_project.data import CorruptMNISTDataModule, corrupt_mnist, normalize, preprocess_data

# Check for existence of the data directory rather than a specific file
data_dir = os.path.join("data", "processed")


@pytest.mark.skipif(not os.path.exists(data_dir), reason="Processed data directory not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Train targets did not have the correct number of samples"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Test targets did not have the correct number of samples"


def test_datamodule_fit_stage(tmp_path):
    dm = CorruptMNISTDataModule(data_dir="data/processed", batch_size=32)
    dm.setup(stage="fit")  # This covers the "fit" branch
    assert dm.train_dataset is not None, "train_dataset not created in fit stage"
    assert dm.test_dataset is not None, "test_dataset not created in fit stage"


def test_datamodule_test_stage(tmp_path):
    dm = CorruptMNISTDataModule(data_dir="data/processed", batch_size=32)
    dm.setup(stage="test")  # This covers the "test" branch
    assert dm.test_dataset is not None, "test_dataset not created in test stage"


@pytest.mark.parametrize("n_files", [6])
def test_preprocess_data(tmp_path, n_files):
    # 1. Create raw_dir with minimal .pt files
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # We expect up to 6 chunked files for train_images_i.pt and train_target_i.pt
    # Just put minimal data so we can see if it processes correctly
    for i in range(n_files):
        torch.save(torch.randn(10, 28, 28), raw_dir / f"train_images_{i}.pt")
        torch.save(torch.randint(0, 10, (10,)), raw_dir / f"train_target_{i}.pt")

    # Also create single test_images and test_target
    torch.save(torch.randn(5, 28, 28), raw_dir / "test_images.pt")
    torch.save(torch.randint(0, 10, (5,)), raw_dir / "test_target.pt")

    # 2. Create processed_dir
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # 3. Call preprocess_data
    preprocess_data(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

    # 4. Check that we have new .pt files in processed_dir
    expected_files = ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
    for fname in expected_files:
        out_f = processed_dir / fname
        assert out_f.exists(), f"{fname} was not created by preprocess_data!"

    # 5. Optionally, check shapes
    train_images = torch.load(processed_dir / "train_images.pt", weights_only=True)
    train_targets = torch.load(processed_dir / "train_target.pt", weights_only=True)
    assert train_images.shape[1] == 1, "Expected images to be unsqueezed (channels=1)."
    assert train_images.dtype == torch.float32, "Expected images to be float after conversion."
    assert train_targets.dtype == torch.long, "Targets should be long."


def test_normalize():
    images = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm_images = normalize(images)
    # Check that mean is approximately 0 and std is 1
    assert torch.isclose(norm_images.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(norm_images.std(), torch.tensor(1.0), atol=1e-5)
