import logging
import fastapi

from uvicorn import Config

import constants
from .common import UvicornServer
from mindctrl.config import AppSettings
from .local import DeploymentServerContainer

_logger = logging.getLogger(__name__)


def create_mock_supervisor(appSettings: AppSettings):
    mockSupervisor = fastapi.FastAPI()

    @mockSupervisor.get("/")
    def read_root():
        return {"Mock": "Supervisor root"}

    @mockSupervisor.get("/info")
    def read_info():
        return {"Mock": "Supervisor info"}

    @mockSupervisor.get("/addons/self/info")
    def read_addon_info():
        # TODO: why do i need this..
        return {"Mock": "Supervisor addon info"}

    @mockSupervisor.get("/supervisor/ping")
    def read_ping():
        return {"Mock": "Supervisor ping"}

    @mockSupervisor.get("/addons/self/options/config")
    def read_config():
        assert appSettings.store.store_type == "psql"
        assert appSettings.events.events_type == "mqtt"
        ev_password = (
            appSettings.events.password.get_secret_value()
            if appSettings.events.password is not None
            else ""
        )
        ev_user = (
            appSettings.events.username
            if appSettings.events.username is not None
            else ""
        )
        config: dict[str, str] = {
            "STORE__STORE_TYPE": appSettings.store.store_type,
            "STORE__USER": appSettings.store.user,
            "STORE__PASSWORD": appSettings.store.password.get_secret_value(),
            "STORE__ADDRESS": "0.0.0.0",  # appSettings.store.address,
            "STORE__PORT": str(appSettings.store.port),
            "STORE__DATABASE": appSettings.store.database,
            "EVENTS__EVENTS_TYPE": appSettings.events.events_type,
            "EVENTS__BROKER": appSettings.events.broker,
            "EVENTS__PORT": str(appSettings.events.port),
            "EVENTS__USERNAME": ev_user,
            "EVENTS__PASSWORD": ev_password,
            "OPENAI_API_KEY": appSettings.openai_api_key.get_secret_value(),
        }

        return {"data": config}

    return UvicornServer(
        Config(mockSupervisor, log_level="debug", host="0.0.0.0", workers=1)
    )


class AddonContainer(DeploymentServerContainer):
    def __init__(
        self,
        supervisor_url: str,
        *args,
        image="addon:latest",
        port=constants.LOCAL_MULTISERVER_PORT,
        **kwargs,
    ):
        super().__init__(*args, image=image, network_mode="host", port=port, **kwargs)
        self.with_env("SUPERVISOR_API", supervisor_url)
