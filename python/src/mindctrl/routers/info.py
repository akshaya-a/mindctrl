import logging
from fastapi import APIRouter, Request
import mlflow

router = APIRouter(tags=["info"])


_logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/version")
def read_version(request: Request):
    ingress_header = request.headers.get("X-Ingress-Path")
    _logger.info("INGRESS PATH:")
    _logger.info(ingress_header)

    ws_url = (
        f"{ingress_header}/ws"
        if ingress_header
        else f"{request.url_for('websocket_endpoint')}"
    )
    ingress_header = ingress_header or ""
    chat_url = f"{ingress_header}/deployed-models/chat/labels/latest/invocations"
    mlflow_url = request.base_url.replace(port=5000)
    dashboard_url = request.base_url.replace(port=9999)

    return {
        "tracking_store": mlflow.get_tracking_uri(),
        "model_registry": mlflow.get_registry_uri(),
        "ws_url": ws_url,
        "chat_url": chat_url,
        "mlflow_url": mlflow_url,
        "dashboard_url": dashboard_url,
    }
