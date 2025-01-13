import pytorch_lightning as pl
import torch
import typer
from torch.utils.data import DataLoader, TensorDataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt", weights_only=True))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt", weights_only=True))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt", weights_only=True)
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt", weights_only=True)

    # Reshape and convert
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist(processed_dir: str = "data/processed") -> tuple[TensorDataset, TensorDataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load(f"{processed_dir}/train_images.pt", weights_only=True)
    train_target = torch.load(f"{processed_dir}/train_target.pt", weights_only=True)
    test_images = torch.load(f"{processed_dir}/test_images.pt", weights_only=True)
    test_target = torch.load(f"{processed_dir}/test_target.pt", weights_only=True)

    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)
    return train_set, test_set


class CorruptMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/processed", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        """Load data, split or transform as needed."""
        if stage == "fit" or stage is None:
            self.train_dataset, self.test_dataset = corrupt_mnist(self.data_dir)
            # Optionally, create a validation dataset split if needed
            # Here we’ll just treat the test_dataset as validation for example’s sake.

        if stage == "test" or stage is None:
            # If you have a separate test dataset, load it
            # but for now, re-use the same test_dataset if needed
            _, self.test_dataset = corrupt_mnist(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # We can just reuse the test_dataset as "validation" here or create a real val split
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    typer.run(preprocess_data)
