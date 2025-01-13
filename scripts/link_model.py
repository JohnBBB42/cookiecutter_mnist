import os
import typer
import wandb


def link_model(
    artifact_path: str = typer.Argument(..., help="Artifact path in the form entity/project/artifact:version"),
    aliases: list[str] = typer.Option(
        ["staging"],  # Default value
        "--alias",  # Long option name
        "-a",  # Short option name
        help="Aliases to link the artifact with. Can be repeated.",
    ),
) -> None:
    """
    Stage a specific model to the model registry.

    Example:
        python link_model.py entity/project/artifact:version -a staging -a production
    """
    if not artifact_path:
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    _, _, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}", aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
