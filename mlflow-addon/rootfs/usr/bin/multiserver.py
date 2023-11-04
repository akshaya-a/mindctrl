import os
from typing import Union

from fastapi import FastAPI
import mlflow

app = FastAPI()

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri is not None:
    mlflow.set_tracking_uri(tracking_uri)


@app.get("/")
def read_root():
    return {
        "Tracking Store": mlflow.get_tracking_uri(),
        "Model Registry": mlflow.get_registry_uri(),
    }


@app.get("/models")
def list_models():
    models = mlflow.search_registered_models()
    return {model.name: model.last_updated_timestamp for model in models}


# Write readiness: https://skarnet.org/software/s6/notifywhenup.html
notification_fd = os.environ.get("NOTIFY_FD")
if notification_fd:
    os.write(int(notification_fd), b"\n")
    os.close(int(notification_fd))
