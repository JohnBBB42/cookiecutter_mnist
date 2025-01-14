# train.py
import pytorch_lightning as pl
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mnist_project.data import CorruptMNISTDataModule
from mnist_project.lightning import MyAwesomeModel

app = typer.Typer()


@app.command()
def main():
    data_module = CorruptMNISTDataModule(data_dir="data/processed", batch_size=64)
    model = MyAwesomeModel()
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        default_root_dir="my_logs_dir",
        max_epochs=10,
        limit_train_batches=0.2,
        callbacks=[early_stopping_callback, checkpoint_callback],
        profiler="simple",
        logger=pl.loggers.WandbLogger(project="lightning_mnist"),
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    app()
