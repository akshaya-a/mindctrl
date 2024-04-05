import logging
import os
from pathlib import Path
import yaml

from mindctrl.config import AppSettings

_logger = logging.getLogger(__name__)


def test_addon_options_map_server_options(monkeypatch):
    config_file = Path(__file__).parent.parent / "mindctrl-addon" / "config.yaml"
    config = yaml.safe_load(config_file.read_text())
    options = config["options"]

    if os.environ.get("STORE__STORE_TYPE"):
        # This is not a clean environment, probably local
        monkeypatch.delenv("STORE__STORE_TYPE", raising=False)
        monkeypatch.delenv("STORE__USER", raising=False)
        monkeypatch.delenv("STORE__PASSWORD", raising=False)
        monkeypatch.delenv("STORE__ADDRESS", raising=False)
        monkeypatch.delenv("STORE__PORT", raising=False)
        monkeypatch.delenv("STORE__DATABASE", raising=False)
        monkeypatch.delenv("EVENTS__EVENTS_TYPE", raising=False)
        monkeypatch.delenv("EVENTS__BROKER", raising=False)
        monkeypatch.delenv("EVENTS__PORT", raising=False)
        monkeypatch.delenv("EVENTS__USERNAME", raising=False)
        monkeypatch.delenv("EVENTS__PASSWORD", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for key, value in options.items():
        monkeypatch.setenv(key, value)

    AppSettings()  # pyright: ignore


def test_addon_options_map_schema():
    config_file = Path(__file__).parent.parent / "mindctrl-addon" / "config.yaml"
    config = yaml.safe_load(config_file.read_text())
    options = config["options"]
    schema = config["schema"]

    assert set(options.keys()) == set(schema.keys())
