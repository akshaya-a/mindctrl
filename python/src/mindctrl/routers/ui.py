import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from mindctrl.const import TEMPLATES_DIR

# from mindctrl.workflows import WorkflowContext

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


# TODO: https://fastapi.tiangolo.com/advanced/websockets/#handling-disconnections-and-multiple-clients
# @router.websocket("/mctrlws")
# async def websocket_endpoint2(websocket: WebSocket):
#     await websocket.accept()
#     _logger.info("Websocket accepted")

#     await websocket.accept()
#     queue = asyncio.queues.Queue()

#     async def read_from_socket(websocket: WebSocket):
#         async for data in websocket.iter_json():
#             print(f"putting {data} in the queue")
#             queue.put_nowait(data)

#     async def get_data_and_send():
#         data = await queue.get()
#         while True:
#             if queue.empty():
#                 print(f"getting weather data for {data}")
#                 await asyncio.sleep(1)
#             else:
#                 data = queue.get_nowait()
#                 print(f"Setting data to {data}")

#     await asyncio.gather(read_from_socket(websocket), get_data_and_send())

#     # If doesn't exist, starts workflow
#     # If exists and is paused, resumes workflow
#     workflow_context: WorkflowContext = websocket.state.workflow_context
#     session_id = await mindctrl.get_or_create_conversation(client_id)

#     await asyncio.sleep(10)
#     ring_buffer = iter(websocket.state.state_ring_buffer.copy())
#     while True:
#         try:
#             payload = next(ring_buffer)
#             await websocket.send_json(payload)
#             await asyncio.sleep(1)

#             message = await websocket.receive_json()
#             _logger.info(f"Message received: {message}")
#             assert message["type"] == "mindctrl.chat.user"
#             chat_message = Message(content=message["content"], role="user")

#             # TODO: Actually expand the polling loop, so send should be quick
#             # Then poll on the assistant response
#             # You might not need a workflow for multiturn?
#             assistant_response = await mindctrl.send_message(session_id, chat_message)
#             await websocket.send_json(assistant_response)

#         except StopIteration:
#             _logger.warning("Websocket buffer empty, waiting for new events")
#             await asyncio.sleep(2)
#             ring_buffer = iter(websocket.state.state_ring_buffer.copy())
#         except WebSocketDisconnect:
#             _logger.warning("Websocket disconnected")

#             # Pauses workflow
#             # Sets a timer to terminate the workflow if no activity for X minutes
#             mindctrl.disconnect_conversation(session_id)

#             break
