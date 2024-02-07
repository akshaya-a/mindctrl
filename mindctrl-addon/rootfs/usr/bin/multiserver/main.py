import logging
import os

# Eventing - move this to plugin
from contextlib import asynccontextmanager
import asyncio

# Core functionality
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import mlflow
from mlflow.utils.proto_json_utils import dataframe_from_parsed_json

import pandas as pd

import collections

from .models import log_system_models, poll_registry, SUMMARIZATION_PROMPT
from .mqtt import subscribe_to_mqtt
from .mlflow import connect_to_mlflow


_logger = logging.getLogger(__name__)


def write_healthcheck_file():
    # Write readiness: https://skarnet.org/software/s6/notifywhenup.html
    notification_fd = os.environ.get("NOTIFY_FD")
    if notification_fd:
        os.write(int(notification_fd), b"\n")
        os.close(int(notification_fd))


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(poll_registry(10.0))

    # The buffer should be enhanced to be token-aware
    state_ring_buffer = collections.deque(maxlen=100)

    client = subscribe_to_mqtt(state_ring_buffer)
    loaded_models = log_system_models()
    connect_to_mlflow()

    write_healthcheck_file()

    # Make resources available to requests via .state
    yield {"state_ring_buffer": state_ring_buffer, "loaded_models": loaded_models}

    client.loop_stop()


app = FastAPI(lifespan=lifespan)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))


# from .rag import extract_timestamps, retrieve_events


# # LLM decide query timerange -> statelookup -> in-mem faiss -> query index
# # OAI functioncalling/guided prompt -> llamaindex "docstore" -> lli
# def invocation_pipeline(request: Request, query: str):
#     range_model = mlflow.pyfunc.load_model(model_uri=f"models:/querytime/latest")
#     query_range_response = range_model.predict(pd.DataFrame({"query": [query]}))
#     start, end = extract_timestamps(query_range_response)
#     all_events = retrieve_events(request, start, end)
#     index = faiss(all_events)
#     # TODO: this only works for summarization/retrieval tasks. What about actions?
#     ## If index is dumb
#     relevant_events = index.query(query)
#     return langchain.run(relevant_events, query)


@app.get("/")
def read_root(request: Request, response_class=HTMLResponse):
    ingress_header = request.headers.get("X-Ingress-Path")
    print("INGRESS PATH:")
    print(ingress_header)

    ws_url = (
        f"{ingress_header}/ws"
        if ingress_header
        else f"{request.url_for('websocket_endpoint')}"
    )
    ingress_header = ingress_header or ""
    chat_url = f"{ingress_header}/deployed-models/chat/labels/latest/invocations"
    mlflow_url = request.base_url.replace(port=5000)
    print(mlflow_url)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tracking_store": mlflow.get_tracking_uri(),
            "model_registry": mlflow.get_registry_uri(),
            "ws_url": ws_url,
            "chat_url": chat_url,
            "mlflow_url": mlflow_url,
        },
    )


@app.get("/version")
def read_version(request: Request):
    ingress_header = request.headers.get("X-Ingress-Path")
    print("INGRESS PATH:")
    print(ingress_header)

    ws_url = (
        f"{ingress_header}/ws"
        if ingress_header
        else f"{request.url_for('websocket_endpoint')}"
    )
    ingress_header = ingress_header or ""
    chat_url = f"{ingress_header}/deployed-models/chat/labels/latest/invocations"
    mlflow_url = request.base_url.replace(port=5000)

    import os

    version = os.environ.get("MINDCTRL_ADDON_VERSION", "0.0.0")
    print(f"Version: {version}")
    return {
        "version": version,
        "tracking_store": mlflow.get_tracking_uri(),
        "model_registry": mlflow.get_registry_uri(),
        "ws_url": ws_url,
        "chat_url": chat_url,
        "mlflow_url": mlflow_url,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _logger.info("Websocket accepted")
    await asyncio.sleep(10)
    ring_buffer = iter(websocket.state.state_ring_buffer.copy())
    while True:
        try:
            payload = next(ring_buffer)
            await websocket.send_json(payload)
            await asyncio.sleep(1)
        except StopIteration:
            _logger.warning("Websocket buffer empty, waiting for new events")
            await asyncio.sleep(10)
            ring_buffer = iter(websocket.state.state_ring_buffer.copy())


# Jobs to be done (by integration)
# - summarize current state buffer (this should be reframed as memory?)
#   -> needs a simple "what's going on" endpoint
# - generate an image for guest dashboard
#   -> needs a simple "generate image" endpoint
# - generate an image for art tv


# This logic is obviously wrong, stub impl
@app.get("/deployed-models")
def list_deployed_models():
    models = mlflow.search_registered_models()
    return {model.name: model.last_updated_timestamp for model in models}


@app.get("/state")
def get_current_state(request: Request):
    return request.state.state_ring_buffer


def generate_state_lines(buffer: collections.deque):
    # TODO: when I get internet see if RAG framework already has a known technique to deal with context chunking
    import tiktoken

    enc = tiktoken.encoding_for_model(
        "gpt-3.5-turbo"
    )  # TODO: pick it up from the model meta
    MAX_TOKENS = 4000  # TODO: Also pick it up from the model meta and encode the slack into a smarter heuristic
    buffer_lines = []
    total_tokens = len(enc.encode(SUMMARIZATION_PROMPT))
    for index, item in enumerate(buffer):
        buffer_line = f"{item}"
        next_tokens = len(enc.encode(buffer_line)) + 1  # \n
        if total_tokens + next_tokens >= MAX_TOKENS:
            _logger.warning(
                f"Only added {index + 1} events to message out of {len(buffer)}"
            )
            break
        buffer_lines.append(buffer_line)
        total_tokens += next_tokens

    state_lines = "\n".join(buffer_lines)
    print(f"Generated {total_tokens} token message data:\n{state_lines}")
    return state_lines


def invoke_model_impl(model, payload: dict, request: Request):
    input = dataframe_from_parsed_json(payload["dataframe_split"], "split")
    input["state_lines"] = generate_state_lines(request.state.state_ring_buffer)
    return model.predict(input)


@app.post("/deployed-models/{model_name}/versions/{model_version}/invocations")
def invoke_model_version(
    model_name: str, model_version: str, payload: dict, request: Request
):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return invoke_model_impl(model, payload, request)


@app.post("/deployed-models/{model_name}/labels/{model_label}/invocations")
def invoke_labeled_model_version(
    model_name: str, model_label: str, payload: dict, request: Request
):
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_label}"
        )
    except mlflow.MlflowException as e:
        _logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading model: {e}")
    return invoke_model_impl(model, payload, request)


@app.post("/deployed-labels/{model_label}/invocations")
def invoke_model_label(model_label: str):
    return f"scored {model_label}"
