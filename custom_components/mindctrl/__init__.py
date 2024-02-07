"""MLflow integration for Home Assistant"""
# started from homeassistant/components/openai_conversation

from __future__ import annotations

from functools import partial
import logging

import mlflow as mlflowlib

import voluptuous as vol

from homeassistant.components import conversation as haconversation
from homeassistant.components.hassio import AddonManager, AddonError, AddonState
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import (
    HomeAssistant,
    SupportsResponse,
    callback
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

import asyncio

from .addon import get_addon_manager

from .const import (
    ADDON_NAME, CONF_URL, CONF_USE_ADDON, DOMAIN, _LOGGER
)

from .services import MindctrlClient, async_register_services
from .conversation import MLflowAgent


CONNECT_TIMEOUT = 10


CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the mindctrl component."""
    hass.data[DOMAIN] = {}
    for entry in hass.config_entries.async_entries(DOMAIN):
        if not isinstance(entry.unique_id, str):
            hass.config_entries.async_update_entry(
                entry, unique_id=str(entry.unique_id)
            )
    return True

async def update_listener(hass, entry):
    """Handle options update."""
    # https://developers.home-assistant.io/docs/config_entries_options_flow_handler#signal-updates
    _LOGGER.error(f"update_listener {entry}")

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MLflow from a config entry."""
    _LOGGER.error("mindctrl async_setup_entry")

    hass.data.setdefault(DOMAIN, {})

    # if use_addon := entry.data.get(CONF_USE_ADDON):
    #     await async_ensure_addon_running(hass, entry)

    # TODO 1. Create API instance
    # TODO 2. Validate the API connection (and authentication)
    # TODO 3. Store an API object for your platforms to access
    # hass.data[DOMAIN][entry.entry_id] = MyApi(...)

    _LOGGER.error(f"entry.data: {entry.data}")
    client = MindctrlClient(hass, entry)
    version = None
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT):
            version = await client.version()
        _LOGGER.error(f"setup_entry version: {version}")
    except (asyncio.TimeoutError,) as err:
        raise ConfigEntryNotReady(f"Failed to connect: {err}") from err

    if not version:
        return False

    _LOGGER.error("Connected to mindctrl server")

    async_register_services(hass, entry.data[CONF_URL])

    _LOGGER.error("async_register_services done, setting up agent...")
    haconversation.async_set_agent(hass, entry, MLflowAgent(hass, entry))

    hass.data[DOMAIN][entry.entry_id] = client
    # TODO: Add when there are autogenned virtual entries (sensors) via platforms
    # await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload MLflow."""
    hass.data[DOMAIN].pop(entry.entry_id)
    haconversation.async_unset_agent(hass, entry)

    if entry.data.get(CONF_USE_ADDON) and entry.disabled_by:
        addon_manager: AddonManager = get_addon_manager(hass)
        _LOGGER.debug(f"Stopping {ADDON_NAME} add-on")
        try:
            await addon_manager.async_stop_addon()
        except AddonError as err:
            _LOGGER.error("Failed to stop the mindctrl add-on: %s", err)
            return False
    return True


async def async_ensure_addon_running(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Ensure that mindctrl add-on is installed and running."""
    addon_manager = _get_addon_manager(hass)
    try:
        addon_info = await addon_manager.async_get_addon_info()
    except AddonError as err:
        raise ConfigEntryNotReady(err) from err

    addon_state = addon_info.state

    addon_config = {
    }

    if addon_state == AddonState.NOT_INSTALLED:
        addon_manager.async_schedule_install_setup_addon(
            addon_config,
            catch_error=True,
        )
        raise ConfigEntryNotReady

    if addon_state == AddonState.NOT_RUNNING:
        addon_manager.async_schedule_setup_addon(
            addon_config,
            catch_error=True,
        )
        raise ConfigEntryNotReady

    addon_options = addon_info.options
    updates = {}
    if updates:
        hass.config_entries.async_update_entry(entry, data={**entry.data, **updates})


@callback
def _get_addon_manager(hass: HomeAssistant) -> AddonManager:
    """Ensure that mindctrl add-on is updated and running."""
    addon_manager: AddonManager = get_addon_manager(hass)
    if addon_manager.task_in_progress():
        raise ConfigEntryNotReady
    return addon_manager
