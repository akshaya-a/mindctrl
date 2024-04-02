from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# this is just to make settings typing happy - I don't have another implementation yet
class UnknownEventsSettings(BaseModel):
    events_type: Literal["unknown"]


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
class UnknownStoreSettings(BaseModel):
    store_type: Literal["unknown"]


class AppSettings(BaseSettings):
    # double underscore, in case your font doesn't make it clear
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    store: Union[PostgresStoreSettings, UnknownStoreSettings] = Field(
        discriminator="store_type"
    )
    events: Union[MqttEventsSettings, UnknownEventsSettings] = Field(
        discriminator="events_type"
    )
    # TODO: move this into the gateway or something
    openai_api_key: SecretStr
    force_publish_models: bool = False
    notify_fd: Optional[int] = None
    include_challenger_models: bool = True
    mlflow_tracking_uri: Optional[str] = None
