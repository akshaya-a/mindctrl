import os
from typing import Union

# Eventing - move this to plugin
from contextlib import asynccontextmanager
import asyncio

# Core functionality
from fastapi import FastAPI
import mlflow
from mlflow.utils.proto_json_utils import dataframe_from_parsed_json


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

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Model Registry URI: {mlflow.get_registry_uri()}")


# Sample model
def sample_model(input_prompt: str) -> str:
    return f"scored: {input_prompt}"


from mlflow.openai import log_model
import openai

log_model(
    model="gpt-3.5-turbo",
    task=openai.ChatCompletion,
    messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
    artifact_path="model",
    registered_model_name="chat",
)
log_model(
    model="text-embedding-ada-002",
    task=openai.Embedding,
    artifact_path="embeddings",
    registered_model_name="embeddings",
)


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


@app.post("/deployed-models/{model_name}/versions/{model_version}/invocations")
def invoke_model_version(model_name: str, model_version: str, payload: dict):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    input = dataframe_from_parsed_json(payload["dataframe_split"], "split")
    return model.predict(input)


@app.post("/deployed-labels/{model_label}/invocations")
def invoke_model_label(model_label: str):
    return f"scored {model_label}"


# Write readiness: https://skarnet.org/software/s6/notifywhenup.html
notification_fd = os.environ.get("NOTIFY_FD")
if notification_fd:
    os.write(int(notification_fd), b"\n")
    os.close(int(notification_fd))
