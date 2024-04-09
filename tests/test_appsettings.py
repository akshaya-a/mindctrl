from pydantic_core import ValidationError
from mindctrl.config import AppSettings

import pytest


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
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "test_uri")
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
    monkeypatch.setenv("STORE__STORE_TYPE", "sqlite")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "mqtt")
    with pytest.raises(
        ValidationError,
        match="Input tag 'sqlite' found using 'store_type' does not match any of the expected tags:",
    ):
        settings = AppSettings()  # pyright: ignore
        print(settings)


def test_invalid_events(monkeypatch):
    monkeypatch.setenv("STORE__STORE_TYPE", "psql")
    monkeypatch.setenv("EVENTS__EVENTS_TYPE", "kafka")
    with pytest.raises(
        ValidationError,
        match="Input tag 'kafka' found using 'events_type' does not match any of the expected tags:",
    ):
        settings = AppSettings()  # pyright: ignore
        print(settings)
