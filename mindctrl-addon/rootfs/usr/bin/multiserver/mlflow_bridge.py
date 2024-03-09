import mlflow
from mlflow import MlflowClient
import os
import logging
import asyncio


_logger = logging.getLogger(__name__)


def connect_to_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    _logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    _logger.info(f"Model Registry URI: {mlflow.get_registry_uri()}")

from .const import CHAMPION_ALIAS, CHALLENGER_ALIAS

def is_deployable_alias(aliases: list[str]) -> bool:
    if not aliases:
        return False
    return CHAMPION_ALIAS in aliases or CHALLENGER_ALIAS in aliases


# TODO: Add webhooks/eventing to MLflow OSS server. AzureML has eventgrid support
# In its absence, we poll the MLflow server for changes to the model registry
async def poll_registry(delay_seconds: float = 10.0):
    while True:
        # Sync any new models by tag/label/all
        # Solve any environment dependencies mismatch or fail
        # TODO: Consider running a separate server for each model to solve the isolation problem
        _logger.debug("Polling registry for changes")
        client = MlflowClient()
        mvs = client.search_model_versions()
        aliased_versions = [mv for mv in mvs if is_deployable_alias(mv.aliases)]
        print(aliased_versions)
        await asyncio.sleep(delay=delay_seconds)
