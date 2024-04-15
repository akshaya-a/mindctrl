from functools import lru_cache, partial
import logging
import os

# Eventing - move this to plugin
from contextlib import asynccontextmanager
import asyncio

# Core functionality
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

import collections

from fastapi.templating import Jinja2Templates
import mlflow


from .mlmodels import log_system_models
from .mqtt import setup_mqtt_client, listen_to_mqtt
from .mlflow_bridge import connect_to_mlflow, poll_registry
from .db.setup import setup_db, insert_summary
from .config import AppSettings
from .routers import deployed_models, info, ui
from .const import ROUTE_PREFIX, TEMPLATES_DIR


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_healthcheck_file(settings: AppSettings):
    # Write readiness: https://skarnet.org/software/s6/notifywhenup.html
    notification_fd = settings.notify_fd
    if notification_fd:
        os.write(int(notification_fd), b"\n")
        os.close(int(notification_fd))


@lru_cache
def get_settings():
    # env vars can populate the settings
    return AppSettings()  # pyright: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_settings = get_settings()
    _logger.info("Starting mindctrl server with settings:")
    _logger.info(app_settings.model_dump())

    asyncio.create_task(poll_registry(10.0))

    # The buffer should be enhanced to be token-aware
    state_ring_buffer: collections.deque[dict] = collections.deque(maxlen=20)
    _logger.info("Setting up DB")
    # TODO: convert to ABC with a common interface
    if not app_settings.store.store_type == "psql":
        raise ValueError(f"unknown store type: {app_settings.store.store_type}")
    engine = await setup_db(app_settings.store)
    insert_summary_partial = partial(
        insert_summary, engine, app_settings.include_challenger_models
    )

    _logger.info("Setting up MQTT")
    if not app_settings.events.events_type == "mqtt":
        raise ValueError(f"unknown events type: {app_settings.events.events_type}")

    mqtt_client = setup_mqtt_client(app_settings.events)
    loop = asyncio.get_event_loop()
    _logger.info("Starting MQTT listener")
    mqtt_listener_task = loop.create_task(
        listen_to_mqtt(mqtt_client, state_ring_buffer, insert_summary_partial)
    )

    _logger.info("Logging models")
    loaded_models = log_system_models(app_settings.force_publish_models)
    connect_to_mlflow(app_settings)

    write_healthcheck_file(app_settings)

    _logger.info("Finished server setup")
    # Make resources available to requests via .state
    yield {
        "state_ring_buffer": state_ring_buffer,
        "loaded_models": loaded_models,
        "database_engine": engine,
    }

    # Cancel the task
    mqtt_listener_task.cancel()
    # Wait for the task to be cancelled
    try:
        await mqtt_listener_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.include_router(deployed_models.router, prefix=ROUTE_PREFIX)
app.include_router(info.router, prefix=ROUTE_PREFIX)
app.include_router(ui.router, prefix=ROUTE_PREFIX)


# TODO: decide if redirects are better
# @app.get("/")
# def read_root():
#     return RedirectResponse(url=f"{ROUTE_PREFIX}/ui/", status_code=302)


templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/")
def read_root(request: Request, response_class=HTMLResponse):
    _logger.info(
        f"root received at {request.url} with {request.base_url} for {request.app} from {request.client}. Full scope: {request.scope}"
    )
    _logger.info("Request headers:")
    _logger.info(request.headers)
    ingress_header = request.headers.get("X-Ingress-Path")
    _logger.info(f"ingress path: {ingress_header}")

    ws_url = (
        f"{ingress_header}/ws"
        if ingress_header
        else f"{request.url_for('websocket_endpoint')}"
    )
    ingress_header = ingress_header or ""
    chat_url = (
        f"{ingress_header}{ROUTE_PREFIX}/deployed-models/chat/labels/latest/invocations"
    )
    # TODO: get it from the mlflow static prefix env var or constant
    mlflow_url = (
        f"{ingress_header}/mlflow/"
        if ingress_header
        else f"{request.base_url}mlflow/"  # // if /mlflow - use urljoin or something better
    )
    _logger.info(f"mlflow url: {mlflow_url}")
    # TODO: fix dashboard ingress and then add this later. Or, just an exposed port not via ingress
    dashboard_url = request.base_url.replace(port=9999)
    _logger.info(f"dapr dashboard: {dashboard_url}")
    _logger.info(f"root_path: {request.scope.get('root_path')}")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "tracking_store": mlflow.get_tracking_uri(),
            "model_registry": mlflow.get_registry_uri(),
            "ws_url": ws_url,
            "chat_url": chat_url,
            "mlflow_url": mlflow_url,
            "dashboard_url": dashboard_url,
        },
    )
