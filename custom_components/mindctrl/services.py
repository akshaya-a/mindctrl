from homeassistant.helpers.aiohttp_client import async_get_clientsession

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.config_entries import ConfigEntry

import mlflow
from .const import DOMAIN, SERVICE_INVOKE_MODEL, _LOGGER, CONF_URL
import voluptuous as vol
from homeassistant.helpers import config_validation as cv, selector


class MindctrlClient(object):
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        self.config_entry = config_entry
        self.uri = config_entry.data[CONF_URL]
        self.session = async_get_clientsession(hass)
        # store (this) object in hass data
        hass.data.setdefault(DOMAIN, {})[self.config_entry.entry_id] = self

    def __str__(self) -> str:
        return f"MindctrlClient(uri={self.uri}, config_entry={self.config_entry})"

    async def version(self) -> dict:
        try:
            result = await self.session.get(f"{self.uri}/version")
            return (await result.json())["version"]
        except Exception as e:
            _LOGGER.error(f"error during version: {e}")
            raise e

    async def validate_uri(self):
        try:
            result = await self.session.get(self.uri)
            _LOGGER.info(f"validate_uri {self.uri}: {result}")
        except Exception as e:
            _LOGGER.error(f"error during validate_uri {self.uri}: {e}")
            raise e

    async def connect(self):
        # TODO: run the health check
        _LOGGER.info(f"connect to {self.uri}")


def async_register_services(hass: HomeAssistant, uri: str) -> None:
    _LOGGER.error("async_register_services")

    async def invoke_model(call: ServiceCall) -> ServiceResponse:
        """Invoke model API."""
        try:
            print("invoke model API")
            # {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'], 'data': [[1, 0.5], [2, 0.75]]}
            # payload = {"dataframe_split": df.to_dict(orient="split")}
            query_df = {
                "columns": ["query"],
                "data": [[call.data["prompt"]]],
                "index": [0],
            }
            print(query_df)
            _LOGGER.error(f"hass data: {str(hass.data[DOMAIN])}")
            _LOGGER.debug(f"hass data debug: {hass.data[DOMAIN]}")
            # client = hass.data[DOMAIN]. # TODO: handle multiple config entries
            # result = await client.session.post(f"{uri}/deployed-models/chat/labels/latest/invocations", json={"dataframe_split": query_df})
            result = await async_get_clientsession(hass).post(
                f"{uri}/deployed-models/{call.data['model']}/labels/{call.data['label']}/invocations",
                json={"dataframe_split": query_df},
            )
            result = await result.json()
            _LOGGER.error(f"result: {result}")
        except mlflow.MlflowException as err:
            raise HomeAssistantError(f"Error invoking model: {err}") from err

        # TODO: hass service response seems to expect type dictionary
        return {"result": result}

    if not hass.services.has_service(DOMAIN, SERVICE_INVOKE_MODEL):
        _LOGGER.error("registering service")
        hass.services.async_register(
            DOMAIN,
            SERVICE_INVOKE_MODEL,
            invoke_model,
            schema=vol.Schema(
                {
                    vol.Required("prompt"): cv.string,
                    vol.Optional("model", default="chat"): cv.string,  # type: ignore
                    vol.Optional("label", default="latest"): cv.string,  # type: ignore
                }
            ),
            supports_response=SupportsResponse.ONLY,
        )
