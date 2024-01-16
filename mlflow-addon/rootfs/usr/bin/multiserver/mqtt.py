import uuid
import json
import os
import logging

import paho.mqtt.client as mqtt


_logger = logging.getLogger(__name__)


def subscribe_to_mqtt(state_ring_buffer):
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
    return client
