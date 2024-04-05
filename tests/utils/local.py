import logging
from pathlib import Path
import os
import shutil

from uvicorn import Config

import constants
from .common import ServiceContainer, UvicornServer

_logger = logging.getLogger(__name__)


class MosquittoContainer(ServiceContainer):
    def __init__(
        self, image="eclipse-mosquitto:latest", port=constants.MQTT_PORT, **kwargs
    ):
        super().__init__(image, port=port, **kwargs)
        self.with_command("mosquitto -c /mosquitto-no-auth.conf")


class DeploymentServerContainer(ServiceContainer):
    def __init__(
        self,
        route_config: Path,
        replays_dir: Path,
        recordings_dir: Path,
        replay_mode: bool,
        image="deployments:latest",
        port=constants.LOCAL_REPLAY_SERVER_PORT,
        **kwargs,
    ):
        super().__init__(image, port=port, **kwargs)
        self.route_config = route_config
        self.replay = replay_mode
        self.replays_dir = replays_dir
        self.recordings_dir = recordings_dir

        self.with_env(
            "OPENAI_API_KEY", "DUMMY" if self.replay else os.environ["OPENAI_API_KEY"]
        )
        _logger.info(
            f"Mapping {self.replays_dir}, {self.recordings_dir}, {self.route_config.parent} as /replays, /recordings, /config"
        )

        container_replay_dir = "/replays"
        self.with_volume_mapping(
            str(self.replays_dir),
            container_replay_dir,
            mode="ro",
        )
        self.with_env("MINDCTRL_REPLAY_DIR", container_replay_dir)

        container_recordings_dir = "/recordings"
        self.with_volume_mapping(
            str(self.recordings_dir),
            container_recordings_dir,
            mode="rw",
        )
        self.with_env("MINDCTRL_RECORDING_DIR", container_recordings_dir)

        container_config_dir = "/config"
        self.with_volume_mapping(
            str(self.replays_dir.parent),
            container_config_dir,
            mode="ro",
        )
        self.with_env(
            "MLFLOW_DEPLOYMENTS_CONFIG", f"{container_config_dir}/route-config.yaml"
        )

        if self.replay:
            assert (self.replays_dir / "replay.json").exists()
            self.with_env("MINDCTRL_CONFIG_REPLAY", "true")

    def copy_recordings(self, target_dir: Path):
        files = list(self.recordings_dir.glob("*"))
        _logger.info(
            f"Copying recordings to {target_dir}: {', '.join(map(str, files))}"
        )
        shutil.copytree(self.recordings_dir, target_dir, dirs_exist_ok=True)


LocalMultiserver = UvicornServer(
    Config(
        "mindctrl.main:app",
        host=constants.LOCAL_MULTISERVER_HOST,
        port=constants.LOCAL_MULTISERVER_PORT,
        log_level="debug",
    )
)