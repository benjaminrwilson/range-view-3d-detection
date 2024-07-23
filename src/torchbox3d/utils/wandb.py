"""WandB utilities."""

import logging
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)


def load_artifact(
    model_name: str, project_name: str, entity: str, version: int
) -> Path:
    """Load an artifact from Wandb.

    Args:
        model_name: Wandb model name.
        project_name: WandB project name.
        entity: WandB entity name.
        version: Model version.

    Returns:
        Artifact path.
    """
    logger.info(f"Using {model_name} for validation.")
    path = Path(f"artifacts/model-{model_name}:v{version}/model.ckpt")
    if not path.exists():
        run = wandb.init()
        artifact = run.use_artifact(
            f"{entity}/{project_name}/model-{model_name}:v{version}",
            type="model",
        )
        artifact.download()
    return path


def pull_wandb_feather(project_name: str, entity: str, tag: str) -> Path:
    """Load an artifact from Wandb.

    Args:
        model_name: Wandb model name.
        project_name: WandB project name.
        entity: WandB entity name.
        tag: WandB tag.

    Returns:
        Artifact path.
    """
    path = f"{entity}/{project_name}/val_results:{tag}"
    run = wandb.init()
    artifact = run.use_artifact(path, type="metrics")
    artifact.download()
    return Path(f"artifacts/val_results:{tag}")
