import os
from typing import Union

# Eventing - move this to plugin
from contextlib import asynccontextmanager
import asyncio

# Core functionality
from fastapi import FastAPI
import mlflow


# TODO: Add webhooks/eventing to MLflow OSS server. AzureML has eventgrid support
# In its absence, we poll the MLflow server for changes to the model registry
async def poll_registry(delay_seconds: float = 10.0):
    while True:
        print("Polling registry for changes")
        await asyncio.sleep(delay=delay_seconds)


@asynccontextmanager
async def registry_syncer(app: FastAPI):
    asyncio.create_task(poll_registry(10.0))
    yield


app = FastAPI(lifespan=registry_syncer)

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri is not None:
    mlflow.set_tracking_uri(tracking_uri)


@app.get("/")
def read_root():
    return {
        "Tracking Store": mlflow.get_tracking_uri(),
        "Model Registry": mlflow.get_registry_uri(),
    }


# This logic is obviously wrong, stub impl
@app.get("/deployed-models")
def list_deployed_models():
    models = mlflow.search_registered_models()
    return {model.name: model.last_updated_timestamp for model in models}


# Write readiness: https://skarnet.org/software/s6/notifywhenup.html
notification_fd = os.environ.get("NOTIFY_FD")
if notification_fd:
    os.write(int(notification_fd), b"\n")
    os.close(int(notification_fd))
