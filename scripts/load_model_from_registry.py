import wandb
import torch
from src.mnist_project.model import MyAwesomeModel

# Initialize a W&B run (no need to specify a project)
run = wandb.init()

# Use the artifact from the specified registry
artifact = run.use_artifact('jhz209-university-of-copenhagen-org/wandb-registry-ML_OPS/corrupted_mnist:v0', type='model')

# Download the artifact to a local directory
artifact_dir = artifact.download("artifacts/corrupted_mnist_v0")

# Initialize the model and load the weights
model = MyAwesomeModel()
model.load_state_dict(torch.load(f"{artifact_dir}/model.pth"))  # Fixed the string formatting

print("Model loaded successfully!")

