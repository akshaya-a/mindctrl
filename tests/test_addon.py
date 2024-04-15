import logging
import os
from pathlib import Path
import pytest
import yaml

from mindctrl.config import AppSettings

_logger = logging.getLogger(__name__)


@pytest.fixture
def config_as_obj(repo_root_dir: Path):
    config_file = repo_root_dir / "mindctrl-addon" / "config.yaml"
    config = yaml.safe_load(config_file.read_text())
    return config


def test_addon_options_map_server_options(monkeypatch, config_as_obj):
    options = config_as_obj["options"]

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


def test_addon_options_map_schema(config_as_obj):
    options = config_as_obj["options"]
    schema = config_as_obj["schema"]

    assert set(options.keys()) == set(schema.keys())


def test_addon_ingress(config_as_obj, repo_root_dir):
    assert config_as_obj["ingress"] is True
    # TODO: this is a bad assert but the command line is fixed in shell scripts right now
    assert config_as_obj["ingress_port"] == 80
    assert "TRAEFIK_ALLOW_IP" in config_as_obj["environment"].keys()
    assert config_as_obj["environment"]["TRAEFIK_ALLOW_IP"] == "172.30.32.2/32"
    assert config_as_obj["environment"]["TRAEFIK_ALLOW_IPV6"] == ""

    traefik_text = (repo_root_dir / "services/ingress/traefik-config.yaml").read_text()
    # just make sure these don't regress in a bunch of moves
    assert "ipAllowList" in traefik_text
    assert (
        "TRAEFIK_ALLOW_IP" in traefik_text
    )  # this doesn't actually test as it's a substring of v6
    assert "TRAEFIK_ALLOW_IPV6" in traefik_text
