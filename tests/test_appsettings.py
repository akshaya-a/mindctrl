import logging
import os
from pathlib import Path
import subprocess

from dapr.conf import settings
from pydantic import SecretStr
import pytest
from pydantic_core import ValidationError
import sqlalchemy

from mindctrl.config import (
    AppSettings,
    DisabledEventsSettings,
    DisabledHomeAssistantSettings,
    DisabledStoreSettings,
    get_settings,
)
from mindctrl.const import CONFIGURATION_KEY, CONFIGURATION_TABLE

_logger = logging.getLogger(__name__)


def test_basic_appsettings(monkeypatch):
    monkeypatch.setenv("STORE__STORE_TYPE", "psql")
    monkeypatch.setenv("STORE__USER", "user")
    monkeypatch.setenv("STORE__PASSWORD", "test_password")
    monkeypatch.setenv("STORE__ADDRESS", "localhost")
    monkeypatch.setenv("STORE__PORT", "5432")
    monkeypatch.setenv("STORE__DATABASE", "mindctrl")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "mqtt")
    monkeypatch.setenv("EVENTS__BROKER", "localhost")
    monkeypatch.setenv("EVENTS__PORT", "1883")
    monkeypatch.setenv("EVENTS__USERNAME", "user")
    monkeypatch.setenv("EVENTS__PASSWORD", "test_password")
    monkeypatch.setenv("HASS__HASS_TYPE", "remote")
    monkeypatch.setenv("HASS__HOST", "test.local")
    monkeypatch.setenv("HASS__PORT", "8123")
    monkeypatch.setenv("HASS__LONG_LIVED_ACCESS_TOKEN", "fake-token")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "test_uri")
    monkeypatch.setenv("DAPR_MODE", "false")
    settings = AppSettings()  # pyright: ignore
    assert settings.store.store_type == "psql"
    assert settings.store.user == "user"
    assert settings.store.password.get_secret_value() == "test_password"
    assert settings.store.address == "localhost"
    assert settings.store.port == 5432
    assert settings.store.database == "mindctrl"
    assert settings.events.events_type == "mqtt"
    assert settings.events.broker == "localhost"
    assert settings.events.port == 1883
    assert settings.events.username == "user"
    assert settings.events.password is not None
    assert settings.events.password.get_secret_value() == "test_password"
    assert settings.openai_api_key.get_secret_value() == "key"
    assert not settings.force_publish_models
    assert settings.notify_fd is None
    assert settings.include_challenger_models
    assert settings.mlflow_tracking_uri == "test_uri"
    assert "test_password" not in f"{settings.model_dump()}"


def test_invalid_store(monkeypatch):
    monkeypatch.setenv("DAPR_MODE", "false")
    monkeypatch.setenv("STORE__STORE_TYPE", "sqlite")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "mqtt")
    with pytest.raises(
        ValidationError,
        match="Input tag 'sqlite' found using 'store_type' does not match any of the expected tags:",
    ):
        settings = AppSettings()  # pyright: ignore
        print(settings)


def test_invalid_events(monkeypatch):
    monkeypatch.setenv("DAPR_MODE", "false")
    monkeypatch.setenv("STORE__STORE_TYPE", "psql")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "kafka")
    with pytest.raises(
        ValidationError,
        match="Input tag 'kafka' found using 'events_type' does not match any of the expected tags:",
    ):
        settings = AppSettings()  # pyright: ignore
        print(settings)


def test_disable_components(monkeypatch):
    monkeypatch.setenv("DAPR_MODE", "false")
    monkeypatch.setenv("STORE__STORE_TYPE", "none")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "none")
    monkeypatch.setenv("HASS__HASS_TYPE", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "test_uri")
    settings = AppSettings()  # pyright: ignore
    assert settings.store.store_type == "none"
    assert settings.events.events_type == "none"
    assert settings.openai_api_key.get_secret_value() == "key"
    assert not settings.force_publish_models
    assert settings.notify_fd is None
    assert settings.include_challenger_models
    assert settings.mlflow_tracking_uri == "test_uri"
    assert "test_password" not in f"{settings.model_dump()}"


# TODO: Unify with test_workflows.py
@pytest.fixture(scope="session")
def dapr_sidecar_with_config(
    tmp_path_factory: pytest.TempPathFactory,
    repo_root_dir: Path,
    request: pytest.FixtureRequest,
    deploy_mode,
    monkeypatch_session,
    # postgres,
    # placement_server,
):
    if deploy_mode.value != "local":
        # This only makes sense for local testing - dapr is initialized
        # in the container/cluster for addon/k8s
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    # driver_conn_str = postgres.get_connection_url()
    # engine = sqlalchemy.create_engine(driver_conn_str)
    # with engine.begin() as connection:
    #     result = connection.execute(sqlalchemy.text("select version()"))
    #     (version,) = result.fetchone()  # pyright: ignore
    #     _logger.info(version)

    #     result = connection.execute(
    #         sqlalchemy.text(f"""
    #             CREATE TABLE IF NOT EXISTS {CONFIGURATION_TABLE} (
    #                 KEY VARCHAR NOT NULL,
    #                 VALUE VARCHAR NOT NULL,
    #                 VERSION VARCHAR NOT NULL,
    #                 METADATA JSON
    #             );""")
    #     )
    # TODO: Set up the trigger later when implementing dynamic config
    # https://docs.dapr.io/reference/components-reference/supported-configuration-stores/postgresql-configuration-store/#set-up-postgresql-as-configuration-store
    # import re

    # conn_str = re.sub(r"\+\w*", "", driver_conn_str)
    # conn_str = re.sub(r"postgresql", "postgres", conn_str)

    components_path = tmp_path_factory.mktemp("components")

    # state_spec = repo_root_dir / "services" / "components" / "configstore.yaml"
    # assert state_spec.exists(), f"state store spec not found at {state_spec}"
    # target_spec = components_path / "configstore.yaml"

    secret_spec = repo_root_dir / "services" / "components" / "secretstore.yaml"
    assert secret_spec.exists(), f"state store spec not found at {secret_spec}"
    target_secret_spec = components_path / "secretstore.yaml"

    with monkeypatch_session.context() as m:
        # m.setenv("ACTOR_STORE_CONNECTION_STRING", f"{conn_str}")
        # with open(state_spec, "r") as f:
        #     content = f.read()
        #     content = os.path.expandvars(content)
        # with open(target_spec, "w") as f:
        #     f.write(content)

        with open(secret_spec, "r") as f:
            content = f.read()
            content = os.path.expandvars(content)
        with open(target_secret_spec, "w") as f:
            f.write(content)
    _logger.info(f"Generated secret store spec at {target_secret_spec}")

    dapr_process = subprocess.Popen(
        [
            "dapr",
            "run",
            "--app-id",
            request.node.name,
            "--dapr-grpc-port",
            str(settings.DAPR_GRPC_PORT),
            "--dapr-http-port",
            str(settings.DAPR_HTTP_PORT),
            # "--log-level",
            # "debug",
            "--resources-path",
            f"{components_path}",
        ]
    )
    # yield dapr_process, engine
    yield dapr_process
    dapr_process.terminate()


def test_dapr_config(dapr_sidecar_with_config, monkeypatch):
    # _, engine = dapr_sidecar_with_config
    with monkeypatch.context() as m:
        m.setenv("DAPR_MODE", "false")
        temp_settings = AppSettings(
            events=DisabledEventsSettings(),
            store=DisabledStoreSettings(),
            hass=DisabledHomeAssistantSettings(),
            openai_api_key=SecretStr("key"),
            mlflow_tracking_uri="test_uri",
        )
        temp_settings_str = temp_settings.model_dump_json()
        temp_settings_str = temp_settings_str.replace(":", r"\:")
        # with engine.begin() as connection:
        #     result = connection.execute(
        #         sqlalchemy.text(f"""
        #             INSERT INTO {CONFIGURATION_TABLE} (KEY, VALUE, VERSION, METADATA)
        #             VALUES ('{CONFIGURATION_KEY}', '{temp_settings_str}', '1', NULL);
        #         """)
        #     )
        m.setenv("mindctrl.appsettings", temp_settings_str)
        settings = get_settings()
