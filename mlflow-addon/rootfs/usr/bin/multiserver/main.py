import logging
import os
from typing import Union
import uuid
import json

# Eventing - move this to plugin
from contextlib import asynccontextmanager
import asyncio

# Core functionality
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import mlflow
from mlflow.utils.proto_json_utils import dataframe_from_parsed_json

import paho.mqtt.client as mqtt

_logger = logging.getLogger(__name__)

import collections


# TODO: Add webhooks/eventing to MLflow OSS server. AzureML has eventgrid support
# In its absence, we poll the MLflow server for changes to the model registry
async def poll_registry(delay_seconds: float = 10.0):
    while True:
        # Sync any new models by tag/label/all
        # Solve any environment dependencies mismatch or fail
        # TODO: Consider running a separate server for each model to solve the isolation problem
        _logger.debug("Polling registry for changes")
        await asyncio.sleep(delay=delay_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(poll_registry(10.0))
    # TODO: consider something like mqttwarn to decouple the mindctrl server
    # from MQTT directly.

    # Initialize MQTT client:
    client_id = mqtt.base62(uuid.uuid4().int, padding=22)
    client = mqtt.Client(client_id=client_id)
    client.enable_logger(_logger.getChild("mqtt"))
    broker = os.environ.get("MQTT_BROKER", "localhost")
    port = int(os.environ.get("MQTT_PORT", 1883))
    username = os.environ.get("MQTT_USERNAME")
    password = os.environ.get("MQTT_PASSWORD")
    if username and password:
        client.username_pw_set(
            username,
            password,
        )

    # The buffer should be enhanced to be token-aware
    state_ring_buffer = collections.deque(maxlen=100)

    # The callback for when the client receives a CONNACK response from the server.
    def on_mqtt_connect(client, userdata, flags, rc):
        _logger.info(f"Connected with result code {rc}")
        if rc != 0:
            _logger.error(f"Connection failed with result code {rc}")
            raise ConnectionError(f"Connection failed with result code {rc}")

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("hass_ak/#")

    # The callback for when a PUBLISH message is received from the server.
    def on_mqtt_message(client, userdata, msg):
        _logger.debug(f"{msg.topic} {msg.payload}")
        data = json.loads(msg.payload.decode("utf-8"))
        event_type = data.get("event_type", None)
        if event_type is None:
            print("NO EVENT TYPE:")
            print(data)
            return

        if "event_data" not in data:
            print("NO EVENT DATA:")
            print(data)
            return

        if event_type == "state_changed":
            if (
                data["event_data"]["entity_id"].startswith("binary_sensor")
                and data["event_data"]["entity_id"] != "binary_sensor.internet"
            ):
                state_ring_buffer.append(data)
        elif event_type == "call_service":
            if data["event_data"]["domain"] != "system_log":
                state_ring_buffer.append(data)
        elif event_type == "automation_triggered":
            state_ring_buffer.append(data)
        elif event_type == "recorder_5min_statistics_generated":
            return
        else:
            print("UNKNOWN EVENT TYPE:")
            print(data)

    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message

    _logger.info(f"Connecting to MQTT broker {username}@{broker}:{port}")
    client.connect(broker, port, 60)
    client.loop_start()

    # Make resources available to requests via .state
    yield {"state_ring_buffer": state_ring_buffer}

    client.loop_stop()


app = FastAPI(lifespan=lifespan)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Model Registry URI: {mlflow.get_registry_uri()}")


from mlflow.openai import log_model
import openai

SUMMARIZATION_PROMPT = """You're an AI assistant for home automation. You're being given the latest set of events from the home automation system. You are to concisely summarize the events relevant to the user's query followed by an explanation of your reasoning.
EXAMPLE SENSOR DATA:
{{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.washer_wash_completed', 'old_state': {{'entity_id': 'binary_sensor.washer_wash_completed', 'state': 'off', 'attributes': {{'friendly_name': 'Washer Wash completed'}}, 'last_changed': '2023-12-23T09:20:07.695950+00:00', 'last_updated': '2023-12-23T09:20:07.695950+00:00', 'context': {{'id': '01HJAZK20FGR2Z9NTCD46XMQEG', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.washer_wash_completed', 'state': 'on', 'attributes': {{'friendly_name': 'Washer Wash completed'}}, 'last_changed': '2023-12-23T09:53:07.724686+00:00', 'last_updated': '2023-12-23T09:53:07.724686+00:00', 'context': {{'id': '01HJB1FFMCT64MM6GKQX55HMKQ', 'parent_id': None, 'user_id': None}}}}}}}}
{{'event_type': 'call_service', 'event_data': {{'domain': 'tts', 'service': 'cloud_say', 'service_data': {{'cache': True, 'entity_id': ['media_player.kitchen_interrupt', 'media_player.master_bedroom_interrupt'], 'message': 'The washer is complete! Move the clothes to the dryer or they gonna get so so so stinky poo poo!!!!'}}}}}}
{{'event_type': 'call_service', 'event_data': {{'domain': 'media_player', 'service': 'play_media', 'service_data': {{'entity_id': ['media_player.kitchen_interrupt', 'media_player.master_bedroom_interrupt'], 'media_content_id': 'media-source://tts/cloud?message=The+washer+is+complete!+Move+the+clothes+to+the+dryer+or+they+gonna+get+so+so+so+stinky+poo+poo!!!!&cache=true', 'media_content_type': 'music', 'announce': True}}}}}}
{{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'old_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'off', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:46:48.945634+00:00', 'last_updated': '2023-12-23T09:46:48.945634+00:00', 'context': {{'id': '01HJB13XQHGYJYCBH1BS9E6JQY', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'on', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:26.786268+00:00', 'last_updated': '2023-12-23T09:53:26.786268+00:00', 'context': {{'id': '01HJB1G282MSCK7H5KDVE5S260', 'parent_id': None, 'user_id': None}}}}}}}}
{{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'old_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'on', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:26.786268+00:00', 'last_updated': '2023-12-23T09:53:26.786268+00:00', 'context': {{'id': '01HJB1G282MSCK7H5KDVE5S260', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'off', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:50.016556+00:00', 'last_updated': '2023-12-23T09:53:50.016556+00:00', 'context': {{'id': '01HJB1GRY0RSE049YV3NPJ6QFC', 'parent_id': None, 'user_id': None}}}}}}}}

EXAMPLE QUERY: "Is the laundry running?"
EXAMPLE OUTPUT: "The washer is complete. You should move the clothes to the dryer. I see the washer completed sensor turned on at 2023-12-23T09:20:07.695950+00:00"

EXAMPLE QUERY: "Is there anyone in the bedroom?"
EXAMPLE OUTPUT: "There is no one in the bedroom. Even though there was recent activity, I see the bedroom motion sensor turned off at 2023-12-23T09:53:50.016556+00:00"

EXAMPLE QUERY: "What rooms have activity recently?"
EXAMPLE OUTPUT: "The bedroom has had activity. I see the bedroom motion sensor turned on at 2023-12-23T09:53:26.786268+00:00"

Remember to be concise and that there could be multiple sequences of events interleaved, so you can output multiple lines.
"""
log_model(
    model="gpt-3.5-turbo",
    task=openai.ChatCompletion,
    messages=[
        {"role": "system", "content": SUMMARIZATION_PROMPT},
        {"role": "user", "content": "SENSOR DATA:\n{state_lines}\n\nQUERY: {query}"},
    ],
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
def read_root(request: Request, response_class=HTMLResponse):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tracking_store": mlflow.get_tracking_uri(),
            "model_registry": mlflow.get_registry_uri(),
            "ws_url": f"{request.url_for('websocket_endpoint')}",
            "chat_url": f"{request.url_for('invoke_labeled_model_version', model_name='chat', model_label='latest')}",
        },
    )


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


def invoke_model_impl(model, payload: dict, request: Request):
    input = dataframe_from_parsed_json(payload["dataframe_split"], "split")
    buffer_lines = [f"{item}" for item in request.state.state_ring_buffer]
    input["state_lines"] = "\n".join(buffer_lines)
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
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_label}")
    return invoke_model_impl(model, payload, request)


@app.post("/deployed-labels/{model_label}/invocations")
def invoke_model_label(model_label: str):
    return f"scored {model_label}"


# Write readiness: https://skarnet.org/software/s6/notifywhenup.html
notification_fd = os.environ.get("NOTIFY_FD")
if notification_fd:
    os.write(int(notification_fd), b"\n")
    os.close(int(notification_fd))
