"""Provide add-on management."""

# Pattern from
# https://github.com/home-assistant/core/blob/dev/homeassistant/components/zwave_js/addon.py
from __future__ import annotations

from homeassistant.components.hassio import AddonManager
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.singleton import singleton

from .const import _LOGGER, ADDON_NAME, ADDON_SLUG, DOMAIN

DATA_ADDON_MANAGER = f"{DOMAIN}_addon_manager"


@singleton(DATA_ADDON_MANAGER)
@callback
def get_addon_manager(hass: HomeAssistant) -> AddonManager:
    """Get the add-on manager."""
    _LOGGER.debug("get_addon_manager")
    return AddonManager(hass, _LOGGER, ADDON_NAME, ADDON_SLUG)
