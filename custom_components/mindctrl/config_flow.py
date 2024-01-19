from typing import Any, Dict, Optional
from homeassistant import config_entries, core
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import voluptuous as vol
from .const import DOMAIN

import logging
_LOGGER = logging.getLogger(__name__)


async def validate_uri(uri: str, hass: core.HomeAssistant) -> None:
    session = async_get_clientsession(hass)
    try:
        result = session.get(uri)
        _LOGGER.info(f"validate_uri {uri}: {result}")
    except Exception as e:
        _LOGGER.error(f"error during validate_uri {uri}: {e}")
        raise e

class MindctrlFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):

    # The schema version of the entries that it creates
    # Home Assistant will call your migrate method if the version changes
    # (this is not implemented yet)
    VERSION = 1

    config_data: Optional[Dict[str, Any]]

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None):
        errors: Dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_uri(user_input["uri"], self.hass)
            except Exception as e:
                errors["base"] = f"{e}"
            if not errors:
                # Input is valid, set data.
                self.config_data = user_input
                # Return the form of the next step.
                return self.async_create_entry(title="Mindctrl Server", data=self.config_data)

        return self.async_show_form(
            step_id="user", data_schema=vol.Schema({vol.Required("uri"): str})
        )
