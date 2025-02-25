import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mnist_project"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py --raw-dir data/raw --processed-dir data/processed",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run("docker build -t train:latest . -f dockerfiles/train.dockerfile", echo=True, pty=not WINDOWS)
    ctx.run("docker build -t api:latest . -f dockerfiles/api.dockerfile", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Assignment commands
@task
def python(ctx: Context) -> None:
    """Check where the Python is located."""
    ctx.run("which python" if os.name != "nt" else "where python")


@task
def git(ctx: Context, message: str) -> None:
    """Git Add, Push and Commit"""
    ctx.run("git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task
def conda(ctx: Context, name: str = "dtu_mlops") -> None:
    ctx.run(f"conda create --name {name} --file environment.yaml", echo=True)
    ctx.run(f"conda activate {name}", echo=True)
    ctx.run("pip install -e .", echo=True)
