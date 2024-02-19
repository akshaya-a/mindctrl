import mlflow
import os
import logging


_logger = logging.getLogger(__name__)


def connect_to_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    _logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    _logger.info(f"Model Registry URI: {mlflow.get_registry_uri()}")
