# main.py
from pytorch_lightning.cli import LightningCLI
from mnist_project.lightning import MyAwesomeModel
from mnist_project.data import CorruptMNISTDataModule

def cli_main():
    # Initialize the CLI with your model and data module.
    LightningCLI(MyAwesomeModel, CorruptMNISTDataModule)

if __name__ == "__main__":
    cli_main()

