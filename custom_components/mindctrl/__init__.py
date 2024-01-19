"""MLflow integration for Home Assistant"""
# started from homeassistant/components/openai_conversation

from __future__ import annotations

from functools import partial
import logging

import mlflow as mlflowlib

import voluptuous as vol

from homeassistant.components import conversation as haconversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import (
    HomeAssistant,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
)

from .services import MindctrlClient, async_register_services
from .conversation import MLflowAgent

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up MLflow."""
    _LOGGER.info("mindctrl async_setup")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MLflow from a config entry."""
    client = MindctrlClient(hass, entry)
    if not client.validate_uri():
        return False

    async_register_services(hass)

    haconversation.async_set_agent(hass, entry, MLflowAgent(hass, entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload MLflow."""
    hass.data[DOMAIN].pop(entry.entry_id)
    haconversation.async_unset_agent(hass, entry)
    return True
