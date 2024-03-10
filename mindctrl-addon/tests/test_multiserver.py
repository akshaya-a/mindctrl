import pandas as pd
import logging
from pathlib import Path
import aiomqtt


_logger = logging.getLogger(__name__)


async def test_read_root(server_client):
    response = await server_client.get("/")
    assert response.status_code == 200
    assert response.content is not None
    assert response.text is not None
    assert "mlflowContent" in response.text


async def test_read_version(server_client):
    response = await server_client.get("/version")
    assert response.status_code == 200
    assert response.content is not None
    version_data: dict = response.json()
    assert "tracking_store" in version_data.keys()
    assert "model_registry" in version_data.keys()
    assert "ws_url" in version_data.keys()
    assert "chat_url" in version_data.keys()
    assert "mlflow_url" in version_data.keys()
    assert "dashboard_url" in version_data.keys()


async def test_summarize(server_client, hosting_settings):
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

    state_ring_file = Path(__file__).parent / "test_data" / "state_ring_buffer.txt"
    with open(state_ring_file, "r") as f:
        state_lines = f.readlines()
        async with aiomqtt.Client(
            hostname=hosting_settings.events.broker, port=hosting_settings.events.port
        ) as client:
            for event in state_lines:
                await client.publish(topic="hass_ak", payload=event.encode("utf-8"))

    response = await server_client.post(
        "/deployed-models/chat/labels/latest/invocations",
        json=payload,
        timeout=120.0,  # TODO: switch to streaming so we don't need silly long timeouts
    )
    _logger.debug(response.json())
    assert response.status_code == 200
    answers = response.json()
    assert len(answers) == len(df)

    for answer in answers:
        print(answer)

    assert "Yes" in answers[0]
    assert "garage" in answers[0]

    # This is super flaky
    assert "garage" in answers[1].lower()

    assert "no" in answers[2].lower()
