import uuid
from mlflow import MlflowClient
import logging

import pytest


_logger = logging.getLogger(__name__)


async def test_read_health(ingress_client):
    response = await ingress_client.get("/mindctrl/v1/health")
    assert response.status_code == 200
    assert response.content is not None
    assert response.text is not None
    assert "status" in response.text
    assert "ok" in response.text


async def test_read_root(ingress_client):
    response = await ingress_client.get("/")
    assert response.status_code == 200
    assert response.content is not None
    assert response.text is not None
    assert "html" in response.text
    assert "iframe" in response.text
    _logger.debug(response.text)


def test_mlflow_ingress(ingress_client):
    mlflow_client = MlflowClient(tracking_uri=str(ingress_client.base_url))
    mvs = mlflow_client.search_model_versions()
    assert len(mvs) > 0

    random_tag_key = uuid.uuid4().hex
    random_tag_value = uuid.uuid4().hex
    _logger.debug(
        f"Setting mlflow registered model tag {random_tag_key} to {random_tag_value}"
    )
    mlflow_client.set_registered_model_tag(
        name="chat", key=random_tag_key, value=random_tag_value
    )

    _logger.debug(
        f"Searching for mlflow registered model with filter: tags.`{random_tag_key}`='{random_tag_value}'"
    )
    rms = mlflow_client.search_registered_models(
        filter_string=f"tags.`{random_tag_key}`='{random_tag_value}'"
    )
    assert len(rms) == 1
    assert rms[0].name == "chat"
    assert rms[0].tags[random_tag_key] == random_tag_value


# pytest ... --dev will run this test and basically halt fixture teardown
# Then you can poke around the environment etc
@pytest.mark.dev
def test_dev(server_client, mqtt_client, mlflow_client):
    # Request every fixture that matters to trigger setup
    print("DEV MODE ENABLED -- HALTING FIXTURE TEARDOWN")
    print(f"Server client: {server_client.base_url}")
    print(f"MQTT client: {mqtt_client._hostname}:{mqtt_client._port}")
    print(f"MLflow client: {mlflow_client.tracking_uri}")
    breakpoint()
