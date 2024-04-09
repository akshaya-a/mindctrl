import json
import logging
from pathlib import Path
import pytest

from fastapi.testclient import TestClient

from mindctrl.replay_server import create_app_from_env


_logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def replay_server_session(
    monkeypatch_session, replay_server_execution_dir, replay_mode, deploy_mode
):
    if deploy_mode.value != "local":
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    # :( figure out a better directory structure or how to import top level conftest things
    _logger.info(
        f"Starting replay server tests in {replay_mode}, comparing {replay_mode.value}"
    )
    if replay_mode.value == "replay":
        replay_server_execution_dir.populate_replays(
            Path(__file__).parent / "test_data"
        )

    with monkeypatch_session.context() as m:
        m.setenv(
            "MLFLOW_DEPLOYMENTS_CONFIG",
            str(replay_server_execution_dir.config_dir / "route-config.yaml"),
        )
        m.setenv(
            "MINDCTRL_REPLAY_PATH",
            str(replay_server_execution_dir.replays_dir)
            if replay_mode.value == "replay"
            else str(replay_server_execution_dir.recordings_dir),
        )
        m.setenv(
            "MINDCTRL_REPLAY", "true" if replay_mode.value == "replay" else "false"
        )
        if replay_mode.value == "replay":
            m.setenv("OPENAI_API_KEY", "DUMMY")
        app = create_app_from_env()
        yield app

        replay_server_execution_dir.save_recordings(Path(__file__).parent / "test_data")


@pytest.fixture
def test_client(replay_server_session, request):
    return TestClient(
        replay_server_session, headers={"x-mctrl-scenario-name": request.node.name}
    )


def test_docs(test_client):
    response = test_client.get("/docs")
    assert response.status_code == 200


def test_list_endpoints(test_client):
    response = test_client.get("/api/2.0/endpoints/")
    assert response.status_code == 200
    assert "endpoints" in response.json().keys()


def test_get_endpoint(test_client):
    response = test_client.get("/api/2.0/endpoints/chat35t")
    assert response.status_code == 200
    _logger.debug(response.json())


def test_bad_verb_invoke_endpoint(test_client):
    # no GET
    response = test_client.get("/api/2.0/endpoints/chat35t/invocations")
    assert response.status_code == 404


def test_bad_request_invoke_endpoint(test_client):
    # no body
    response = test_client.post("/endpoints/chat35t/invocations")
    assert response.status_code == 422
    # don't worry too much about the structure
    _logger.debug(response.json())
    assert "body" in str(response.json())
    assert "Field required" in str(response.json())


def test_empty_request_body_invoke_endpoint(test_client):
    # bad body
    response = test_client.post("/endpoints/chat35t/invocations", json={})
    assert response.status_code == 422
    _logger.debug(response.json())
    # don't worry too much about the structure
    assert "messages" in str(response.json())
    assert "Field required" in str(response.json())


def test_bad_request_body_invoke_endpoint(test_client):
    # bad body
    response = test_client.post(
        "/endpoints/chat35t/invocations", json={"random": "data"}
    )
    assert response.status_code == 422
    # don't worry too much about the structure
    _logger.debug(response.json())
    assert "messages" in str(response.json())
    assert "Field required" in str(response.json())


def test_basic_chat(test_client):
    response = test_client.post(
        "/endpoints/chat35t/invocations",
        json={"messages": [{"content": "hello", "role": "user"}]},
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] is not None
    assert not str.isspace(message["content"])
    assert message["role"] == "assistant"
    assert message["tool_calls"] is None


def test_tool_call(test_client):
    response = test_client.post(
        "/endpoints/chat35t/invocations",
        json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "what is the weather in Seattle?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    assert response.status_code == 200
    print(response.json())
    message = response.json()["choices"][0]["message"]
    assert (
        message["content"] is None
        or str.isspace(message["content"])
        or len(message["content"]) == 0
    )
    assert message["tool_calls"] is not None
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"]["name"] == "get_current_weather"
    assert message["tool_calls"][0]["function"]["arguments"] is not None
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["location"] == "Seattle"
