import collections
import json
import logging
from typing import Callable, Awaitable, Optional
import aiomqtt
import asyncio

from config import MqttEventsSettings


_logger = logging.getLogger(__name__)


ON_SUMMARY = Callable[[collections.deque], Awaitable[None]]


async def listen_to_mqtt(
    client: aiomqtt.Client,
    state_ring_buffer: collections.deque[dict],
    on_summary: ON_SUMMARY,
    summary_interval: Optional[int] = None,
):
    interval = 5  # Seconds
    event_index = 0
    if summary_interval is None:
        summary_interval = state_ring_buffer.maxlen or 20
    # This convoluted construction is to handle connection errors as the client context needs to be re-entered
    while True:
        try:
            print(
                f"Connecting as {client.identifier} to MQTT broker {client._hostname}:{client._port}..."
            )
            async with client:
                topic = "hass_ak/#"
                print("Connected to MQTT Broker, subscribing to topics ...")
                await client.subscribe(topic=topic)
                print(f"Subscribed to topic {topic}")
                async for msg in client.messages:
                    _logger.debug(f"{msg.topic} {msg.payload}")
                    if not isinstance(msg.payload, bytes):
                        _logger.warning(f"Message payload is not bytes: {msg.payload}")
                        continue
                    try:
                        data: dict = json.loads(msg.payload.decode("utf-8"))
                    except json.JSONDecodeError:
                        print(f"UNDECODABLE MESSAGE:\n{msg.payload.decode('utf-8')}")
                        continue

                    event_type = data.get("event_type", None)
                    if event_type is None:
                        _logger.warning(f"NO EVENT TYPE:\n{data}")
                        continue

                    if "event_data" not in data:
                        _logger.warning(f"NO EVENT TYPE:\n{data}")
                        continue

                    current_count = len(state_ring_buffer)
                    if event_type == "state_changed":
                        if (
                            data["event_data"]["entity_id"].startswith("binary_sensor")
                            and data["event_data"]["entity_id"]
                            != "binary_sensor.internet"
                        ):
                            state_ring_buffer.append(data)
                        # Too noisy with numeric sensors right now
                        # state_ring_buffer.append(data)
                    elif event_type == "call_service":
                        if data["event_data"]["domain"] != "system_log":
                            state_ring_buffer.append(data)
                    elif event_type == "automation_triggered":
                        state_ring_buffer.append(data)
                    elif event_type == "recorder_5min_statistics_generated":
                        continue
                    else:
                        _logger.warning(f"UNKNOWN EVENT TYPE:\n{data}")
                        # TODO: Next step make this more robust with pydantic models
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'script_started', 'event_data': {'name': 'Wake Up', 'entity_id': 'script.wake_up'}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'service_removed', 'event_data': {'domain': 'notify', 'service': 'alexa_media_last_called'}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'service_registered', 'event_data': {'domain': 'notify', 'service': 'alexa_media_last_called'}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'entity_registry_updated', 'event_data': {'action': 'update', 'entity_id': 'media_player.sony_xr_85x90l_2', 'changes': {'capabilities': {}}}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'recorder_hourly_statistics_generated', 'event_data': {}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'config_entry_discovered', 'event_data': {}}
                        # WARNING:multiserver.mqtt:UNKNOWN EVENT TYPE:
                        # {'event_type': 'hue_event', 'event_data': {'id': 'main_bathroom_light_dial_button', 'device_id': '76ba15661915ae06765cf5c64f449fcc', 'unique_id': '5fd10e90-40b4-48d5-9caf-a0116404887d', 'type': 'short_release', 'subtype': 1}}

                    # Summarize the event and write it to storage
                    # https://docs.timescale.com/self-hosted/latest/configuration/telemetry/
                    # https://www.timescale.com/learn/postgresql-extensions-pgvector
                    new_count = len(state_ring_buffer)
                    if new_count > current_count:
                        event_index += new_count - current_count

                    if event_index % summary_interval == 0 and event_index > 0:
                        print(f"Summarizing {event_index} events")
                        await on_summary(state_ring_buffer)
                        event_index = 0

                    continue
        except aiomqtt.MqttError as e:
            _logger.warning(f"{e}\nConnection lost; disposing client ...")
            # This exit should not be needed, but something is holding the client._lock
            await client.__aexit__(None, None, None)
            _logger.warning(f"Reconnecting in {interval} seconds ...")
            await asyncio.sleep(interval)


def setup_mqtt_client(settings: MqttEventsSettings) -> aiomqtt.Client:
    if settings.username and settings.password:
        user = settings.username
        password = settings.password.get_secret_value()
    client = aiomqtt.Client(
        hostname=settings.broker,
        port=settings.port,
        username=user,
        password=password,
        logger=_logger.getChild("mqtt_client"),
        keepalive=60,
    )
    return client
