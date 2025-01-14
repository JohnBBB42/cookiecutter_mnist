import pytest
import torch

from mnist_project.lightning import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


def test_training_step():
    model = MyAwesomeModel()
    # Dummy batch: (images, targets)
    model.log = lambda *args, **kwargs: None
    images = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    batch = (images, targets)

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None, "training_step did not return a loss."
    assert loss.requires_grad, "loss should require grad for backprop."


def test_validation_step():
    model = MyAwesomeModel()
    model.log = lambda *args, **kwargs: None
    images = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    batch = (images, targets)

    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None, "validation_step did not return a loss."


def test_test_step():
    model = MyAwesomeModel()
    model.log = lambda *args, **kwargs: None
    images = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    batch = (images, targets)

    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None, "test_step did not return a loss."
