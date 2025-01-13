import os
import time
import torch
import wandb

# Import the model from your actual module
from mnist_project.lightning import MyAwesomeModel

def load_model(artifact_name: str):
    """
    Load a model from a wandb artifact given the artifact name.
    Example artifact_name: "entity/project/model_name:version"
    """
    # Define a folder where you'll download the artifact
    logdir = "artifacts_download"

    # Initialize wandb API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    # Retrieve the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=logdir)

    # We'll assume there's a single checkpoint file in the artifact
    # If you have multiple files, pick the right one here
    file_name = artifact.files()[0].name

    # Load the model checkpoint
    checkpoint_path = os.path.join(artifact_dir, file_name)
    model = MyAwesomeModel.load_from_checkpoint(checkpoint_path)
    return model

def test_model_speed():
    """
    Test that the model can do 100 predictions in under 1 second.
    MODEL_NAME should be something like 'entity/project/my_model:version'
    """
    # This is the artifact path from environment variable
    artifact_name = os.getenv("MODEL_NAME", "")
    if not artifact_name:
        raise ValueError("MODEL_NAME env variable not set or empty.")

    # Load the model
    model = load_model(artifact_name)
    model.eval()

    # Perform 100 predictions on random input
    start_time = time.time()
    for _ in range(100):
        # Ensure the input is on the correct device if necessary
        inputs = torch.rand(1, 1, 28, 28)
        _ = model(inputs)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time for 100 predictions: {total_time} seconds")

    # Adjust threshold as needed
    assert total_time < 1, "Model took too long to process 100 predictions!"

