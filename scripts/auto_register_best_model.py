import logging
import operator
import os

import typer
from dotenv import load_dotenv

import wandb

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


def stage_best_model_to_registry(model_name: str, metric_name: str = "accuracy", higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        model_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.
    """
    logger.info("Starting to stage the best model...")

    # Initialize the W&B API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    logger.info(f"Connected to W&B entity: {os.getenv('WANDB_ENTITY')}, project: {os.getenv('WANDB_PROJECT')}")

    try:
        artifact_collection = api.artifact_collection(type_name="model", name=model_name)
    except Exception as e:
        logger.error(f"Failed to retrieve artifact collection '{model_name}'. Error: {e}")
        return

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None

    # Process each artifact in the collection
    for artifact in artifact_collection.artifacts():
        logger.info(f"Processing artifact: {artifact.name}")
        logger.info(f"Metadata: {artifact.metadata}")

        if metric_name in artifact.metadata:
            metric_value = artifact.metadata[metric_name]
            logger.info(f"Found {metric_name}: {metric_value}")
            if compare_op(metric_value, best_metric):
                best_metric = metric_value
                best_artifact = artifact
                logger.info(f"New best model found: {artifact.name} with {metric_name}={metric_value}")
        else:
            logger.warning(f"Artifact '{artifact.name}' does not contain metric '{metric_name}'.")

    if best_artifact is None:
        logger.error("No suitable model found in the registry. Exiting...")
        return

    # Stage the best artifact to the registry
    logger.info(f"Staging best model: {best_artifact.name} with {metric_name}={best_metric}")
    try:
        best_artifact.link(
            target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{model_name}",
            aliases=["best", "staging"],
        )
        best_artifact.save()
        logger.info("Best model successfully staged to the registry.")
    except Exception as e:
        logger.error(f"Failed to stage the model. Error: {e}")


if __name__ == "__main__":
    typer.run(stage_best_model_to_registry)
