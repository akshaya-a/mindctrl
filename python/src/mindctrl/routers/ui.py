import asyncio
import logging

from fastapi import APIRouter, WebSocket
from fastapi.templating import Jinja2Templates


from mindctrl.const import TEMPLATES_DIR

router = APIRouter(prefix="/ui", tags=["info"])


_logger = logging.getLogger(__name__)


templates = Jinja2Templates(directory=TEMPLATES_DIR)


# @router.get("/")
# def read_root(request: Request, response_class=HTMLResponse):
#     ingress_header = request.headers.get("X-Ingress-Path")
#     _logger.info(f"ingress path: {ingress_header}")

#     ws_url = (
#         f"{ingress_header}/ws"
#         if ingress_header
#         else f"{request.url_for('websocket_endpoint')}"
#     )
#     ingress_header = ingress_header or ""
#     chat_url = f"{ingress_header}/deployed-models/chat/labels/latest/invocations"
#     mlflow_url = request.base_url.replace(port=5000)
#     _logger.info(f"mlflow url: {mlflow_url}")
#     mlflow_url = f"{mlflow_url}/mlflow"
#     _logger.info(f"mlflow url: {mlflow_url}")
#     dashboard_url = request.base_url.replace(port=9999)
#     _logger.info(f"dapr dashboard: {dashboard_url}")
#     _logger.info(f"root_path: {request.scope.get('root_path')}")

#     return templates.TemplateResponse(
#         request=request,
#         name="index.html",
#         context={
#             "request": request,
#             "tracking_store": mlflow.get_tracking_uri(),
#             "model_registry": mlflow.get_registry_uri(),
#             "ws_url": ws_url,
#             "chat_url": chat_url,
#             "mlflow_url": mlflow_url,
#             "dashboard_url": dashboard_url,
#         },
#     )


@router.websocket("/ws")
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
            await asyncio.sleep(2)
            ring_buffer = iter(websocket.state.state_ring_buffer.copy())
