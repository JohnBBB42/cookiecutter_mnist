import wandb

api = wandb.Api()
artifact_path = "<entity>/<project>/<artifact_name>:<version>"
artifact = api.artifact(artifact_path)
artifact.link(target_path="<entity>/model-registry/<my_registry_name>")
artifact.save()
