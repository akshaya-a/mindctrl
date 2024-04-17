import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates


from mindctrl.const import TEMPLATES_DIR

router = APIRouter(prefix="/ui", tags=["info"])


_logger = logging.getLogger(__name__)


templates = Jinja2Templates(directory=TEMPLATES_DIR)


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
        except WebSocketDisconnect:
            _logger.warning("Websocket disconnected")
            break
