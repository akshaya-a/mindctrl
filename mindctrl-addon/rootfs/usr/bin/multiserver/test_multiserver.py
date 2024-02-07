from fastapi.testclient import TestClient
import pandas as pd
import logging

from .main import app

import pytest
import time
from dotenv import load_dotenv

import paho.mqtt.client as mqtt
import uuid
import os

_logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


def test_read_root(mosquitto, monkeypatch):
    monkeypatch.setenv("MQTT_BROKER", "localhost")
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert list(response.json().keys()) == ["Tracking Store", "Model Registry"]


def test_score_model(mosquitto, monkeypatch):
    df = pd.DataFrame({"animal": ["cats", "dogs"]})
    payload = {"dataframe_split": df.to_dict(orient="split")}
    monkeypatch.setenv("MQTT_BROKER", "localhost")

    with TestClient(app) as client:
        response = client.post(
            "/deployed-models/chat/versions/1/invocations",
            json=payload,
        )
        assert response.status_code == 200
        jokes = response.json()
        assert len(jokes) == 2
        assert "cat" in jokes[0]
        assert "dog" in jokes[1]


def test_summarize(mosquitto, monkeypatch):
    df = pd.DataFrame(
        {
            "query": [
                "Has there been any motion near the garage door?",
                "What rooms have activity?",
                "Should any room have its lights on right now? If so, which ones and why?",
            ]
        }
    )
    payload = {"dataframe_split": df.to_dict(orient="split")}
    monkeypatch.setenv("MQTT_BROKER", "localhost")

    import paho.mqtt.publish as publish

    msgs = []

    from pathlib import Path

    state_ring_file = Path(__file__).parent / "test_data" / "state_ring_buffer.txt"
    with open(state_ring_file, "r") as f:
        state_lines = f.readlines()
        for event in state_lines:
            msgs.append({"topic": "hass_ak", "payload": event.encode("utf-8")})

    publish.multiple(msgs)

    with TestClient(app) as client:
        # state_len = 0
        # while state_len < 20:
        #     time.sleep(5)
        #     response = client.get("/state")
        #     assert response.status_code == 200
        #     state_len = len(response.json())
        #     _logger.debug(f"Current state buffer length: {state_len}")

        response = client.post(
            "/deployed-models/chat/labels/latest/invocations",
            json=payload,
        )
        _logger.debug(response.json())
        assert response.status_code == 200
        answers = response.json()
        assert len(answers) == len(df)
        assert "garage" in answers[0]
        assert "bedroom" in answers[1]
        assert "motion sensor" in answers[1]
        assert "no room" in answers[2].lower()
        for answer in answers:
            print(answer)
