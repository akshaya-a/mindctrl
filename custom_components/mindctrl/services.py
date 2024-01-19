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
from .const import DOMAIN, SERVICE_INVOKE_MODEL
import voluptuous as vol
from homeassistant.helpers import config_validation as cv, selector

import logging
_LOGGER = logging.getLogger(__name__)


class MindctrlClient(object):
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        self.config_entry = config_entry
        self.uri = config_entry.data["uri"]
        self.session = async_get_clientsession(hass)
        # store (this) object in hass data
        hass.data.setdefault(DOMAIN, {})[self.config_entry.entry_id] = self

    def validate_uri(self):
        try:
            result = self.session.get(self.uri)
            _LOGGER.info(f"validate_uri {self.uri}: {result}")
        except Exception as e:
            _LOGGER.error(f"error during validate_uri {self.uri}: {e}")
            raise e

def async_register_services(hass: HomeAssistant) -> None:
    async def invoke_model(call: ServiceCall) -> ServiceResponse:
        """ Invoke model API. """
        try:
            print("invoke model API")
            # {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'], 'data': [[1, 0.5], [2, 0.75]]}
            # payload = {"dataframe_split": df.to_dict(orient="split")}
            query_df = {"columns": ["query"], "data": [[call.data["prompt"]]], "index": [0]}
            print(query_df)
            client = hass.data[DOMAIN].first() # TODO: handle multiple config entries
            result = client.session.post(f"{client.uri}/deployed-models/chat/labels/latest/invocations", json={"dataframe_split": query_df})
        except mlflow.MlflowException as err:
            raise HomeAssistantError(f"Error invoking model: {err}") from err

        return result

    if not hass.services.has_service(DOMAIN, SERVICE_INVOKE_MODEL):
        hass.services.async_register(
            DOMAIN,
            SERVICE_INVOKE_MODEL,
            invoke_model,
            schema=vol.Schema(
                {
                    vol.Required("config_entry"): selector.ConfigEntrySelector(
                        {
                            "integration": DOMAIN,
                        }
                    ),
                    vol.Required("prompt"): cv.string,
                }
            ),
            supports_response=SupportsResponse.ONLY,
        )
