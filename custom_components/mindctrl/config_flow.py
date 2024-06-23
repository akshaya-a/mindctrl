import asyncio
import uuid
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries, exceptions
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant, callback

# from homeassistant.components.zeroconf import ZeroconfServiceInfo
from homeassistant.data_entry_flow import (
    FlowResult,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    _LOGGER,
    ADDON_HOST,
    ADDON_PORT,
    CONF_ADDON_LOG_LEVEL,
    CONF_INTEGRATION_CREATED_ADDON,
    CONF_USE_ADDON,
    DOMAIN,
)

DEFAULT_URL = f"http://{ADDON_HOST}:{ADDON_PORT}"
TITLE = "mindctrl"

ADDON_SETUP_TIMEOUT = 5
ADDON_SETUP_TIMEOUT_ROUNDS = 40
CONF_LOG_LEVEL = "log_level"
SERVER_VERSION_TIMEOUT = 10


class VersionInfo(object):
    def __init__(self, version: str):
        self.version = version
        self.home_id = str(uuid.uuid4())


ADDON_LOG_LEVELS = {
    "error": "Error",
    "warn": "Warn",
    "info": "Info",
    "verbose": "Verbose",
    "debug": "Debug",
    "silly": "Silly",
}
ADDON_USER_INPUT_MAP = {CONF_ADDON_LOG_LEVEL: CONF_LOG_LEVEL}
ON_SUPERVISOR_SCHEMA = vol.Schema(
    {
        vol.Optional(
            CONF_USE_ADDON,
            msg=CONF_USE_ADDON,
            default=True,  # type: ignore
            description="Use the mindctrl addon",
        ): bool
    }
)


def get_manual_schema(user_input: dict[str, Any]) -> vol.Schema:
    """Return a schema for the manual step."""
    default_url = user_input.get(CONF_URL, DEFAULT_URL)
    return vol.Schema({vol.Required(CONF_URL, default=default_url): str})


async def validate_input(hass: HomeAssistant, user_input: dict) -> VersionInfo:
    """Validate if the user input allows us to connect."""
    address = user_input[CONF_URL]

    if not address.startswith(("http://", "https://")):
        raise InvalidInput("invalid_url")

    try:
        return await async_get_version_info(hass, address)
    except CannotConnect as err:
        raise InvalidInput("cannot_connect") from err


async def async_get_version_info(hass: HomeAssistant, address: str) -> VersionInfo:
    """Return mindctrl addon version info."""
    try:
        async with asyncio.timeout(SERVER_VERSION_TIMEOUT):
            print("async_get_version_info")
            version_response = await async_get_clientsession(hass).get(
                f"{address}/version"
            )
            version_info: VersionInfo = (await version_response.json())["version"]
            _LOGGER.error(f"version_info: {version_info}")
    except (asyncio.TimeoutError, aiohttp.ClientError) as err:
        # We don't want to spam the log if the add-on isn't started
        # or takes a long time to start.
        _LOGGER.error("Failed to connect to mindctrl server: %s", err)
        raise CannotConnect from err

    return version_info


async def validate_uri(uri: str, hass: HomeAssistant) -> None:
    session = async_get_clientsession(hass)
    try:
        result = await session.get(uri)
        _LOGGER.info(f"validate_uri {uri}: {result}")
    except Exception as e:
        _LOGGER.error(f"error during validate_uri {uri}: {e}")
        raise e


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for mindctrl."""

    VERSION = 1

    def __init__(self) -> None:
        """Set up flow instance."""
        super().__init__()
        self.use_addon = True

    @property
    def flow_manager(self) -> config_entries.ConfigEntriesFlowManager:
        """Return the correct flow manager."""
        return self.hass.config_entries.flow

    # @staticmethod
    # @callback
    # def async_get_options_flow(
    #     config_entry: config_entries.ConfigEntry,
    # ) -> OptionsFlowHandler:
    #     """Return the options flow."""
    #     return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        _LOGGER.error(f"DEBUG: async_step_user, {user_input}")
        # TODO: discovery seems to be blocked on an allowlist of services:
        # https://github.com/home-assistant/supervisor/tree/main/supervisor/discovery/services
        # if is_hassio(self.hass):
        #     return await self.async_step_on_supervisor()

        return await self.async_step_manual()

    async def async_step_manual(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle a manual configuration."""
        _LOGGER.error(f"DEBUG: async_step_manual, {user_input}")
        if user_input is None:
            return self.async_show_form(
                step_id="manual", data_schema=get_manual_schema({})
            )

        errors = {}

        try:
            _ = await validate_input(self.hass, user_input)
        except InvalidInput as err:
            errors["base"] = err.error
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            await self._async_handle_discovery_without_unique_id()
            self.address = user_input[CONF_URL]
            return self._async_create_entry_from_vars()

        return self.async_show_form(
            step_id="manual", data_schema=get_manual_schema(user_input), errors=errors
        )

    @callback
    def _async_create_entry_from_vars(self) -> FlowResult:
        """Return a config entry for the flow."""
        # Abort any other flows that may be in progress
        _LOGGER.error(f"DEBUG: _async_create_entry_from_vars, {self.address}")
        for progress in self._async_in_progress():
            self.hass.config_entries.flow.async_abort(progress["flow_id"])

        return self.async_create_entry(
            title=TITLE,
            data={
                CONF_URL: self.address,
                CONF_USE_ADDON: self.use_addon,
                # CONF_INTEGRATION_CREATED_ADDON: self.integration_created_addon,
                CONF_INTEGRATION_CREATED_ADDON: False,
            },
        )


class CannotConnect(exceptions.HomeAssistantError):
    """Indicate connection error."""


class InvalidInput(exceptions.HomeAssistantError):
    """Error to indicate input data is invalid."""

    def __init__(self, error: str) -> None:
        """Initialize error."""
        super().__init__()
        self.error = error
