from functools import lru_cache
import json
import logging
import os
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from mindctrl.const import CONFIGURATION_KEY, CONFIGURATION_STORE, SECRET_STORE


_logger = logging.getLogger(__name__)


# this is just to make settings typing happy - I don't have another implementation yet
class DisabledEventsSettings(BaseModel):
    events_type: Literal["none"] = "none"


class MqttEventsSettings(BaseModel):
    events_type: Literal["mqtt"]

    broker: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[SecretStr] = None


class PostgresStoreSettings(BaseModel):
    store_type: Literal["psql"]

    user: str
    password: SecretStr
    address: str = "localhost"
    port: int = 5432
    database: str = "mindctrl"


# Just to make typing happy for now - add dapr, sqlite, etc
class DisabledStoreSettings(BaseModel):
    store_type: Literal["none"] = "none"


class DisabledHomeAssistantSettings(BaseModel):
    hass_type: Literal["none"] = "none"


class SupervisedHomeAssistantSettings(BaseModel):
    hass_type: Literal["supervised"]

    supervisor_token: SecretStr


class RemoteHomeAssistantSettings(BaseModel):
    hass_type: Literal["remote"]

    host: str
    port: int
    long_lived_access_token: SecretStr


def has_dapr() -> bool:
    # TODO: make the request to ensure dapr is running
    # This is really more a hedge because dapr is new
    # and I might hit something I can't work around
    # (but so far the workarounds have... worked around)
    return os.environ.get("DAPR_MODE") != "false"


class AppSettings(BaseSettings):
    # double underscore, in case your font doesn't make it clear
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    store: Union[PostgresStoreSettings, DisabledStoreSettings] = Field(
        discriminator="store_type",
        default=DisabledStoreSettings(),
    )
    events: Union[MqttEventsSettings, DisabledEventsSettings] = Field(
        discriminator="events_type",
        default=DisabledEventsSettings(),
    )
    hass: Union[
        DisabledHomeAssistantSettings,
        SupervisedHomeAssistantSettings,
        RemoteHomeAssistantSettings,
    ] = Field(discriminator="hass_type", default=DisabledHomeAssistantSettings())

    # TODO: move this into the gateway or something
    openai_api_key: SecretStr
    force_publish_models: bool = False
    notify_fd: Optional[int] = None
    include_challenger_models: bool = True
    mlflow_tracking_uri: Optional[str] = None


@lru_cache
def get_settings(**kwargs):
    if has_dapr():
        from dapr.clients import DaprClient  # for typing

        with DaprClient() as dapr_client:
            secret_response = dapr_client.get_secret(SECRET_STORE, CONFIGURATION_KEY)
            print(secret_response)
            return AppSettings.model_validate_json(
                secret_response.secret[CONFIGURATION_KEY]
            )
    # env vars can populate the settings
    return AppSettings()  # pyright: ignore
