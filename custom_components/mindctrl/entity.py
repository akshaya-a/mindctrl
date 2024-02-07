"""AdGuard Home base entity."""
from __future__ import annotations
from abc import ABC, abstractmethod

from homeassistant.config_entries import SOURCE_HASSIO, ConfigEntry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity

from .const import ADDON_SLUG, DOMAIN, _LOGGER
from .services import MindctrlClient

# https://github.com/home-assistant/core/blob/52d27230bce239017722d8ce9dd6f5386f63aba2/homeassistant/components/adguard/entity.py
class MindctrlEntity(Entity, ABC):
    """Defines a base Mindctrl entity."""

    _attr_has_entity_name = True
    _attr_available = True

    def __init__(
        self,
        mindctrl_client: MindctrlClient,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the Mindctrl entity."""
        self._entry = entry
        self.mindctrl_client = mindctrl_client

    async def async_update(self) -> None:
        """Update Mindctrl entity."""
        if not self.enabled:
            return

        try:
            await self._mindctrl_update()
            self._attr_available = True
        except Exception:
            if self._attr_available:
                _LOGGER.error(
                    "An error occurred while updating Mindctrl sensor",
                    exc_info=True,
                )
            self._attr_available = False

    @abstractmethod
    async def _mindctrl_update(self) -> None:
        """Update Mindctrl entity."""
        raise NotImplementedError()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information about this AdGuard Home instance."""
        if self._entry.source == SOURCE_HASSIO:
            config_url = f"homeassistant://hassio/ingress/{ADDON_SLUG}"
        # elif self.mindctrl_client.tls:
        #     config_url = f"https://{self.adguard.host}:{self.adguard.port}"
        else:
            config_url = self.mindctrl_client.uri

        return DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={
                (  # type: ignore[arg-type]
                    DOMAIN,
                    self.adguard.host,
                    self.adguard.port,
                    self.adguard.base_path,
                )
            },
            manufacturer="AK",
            name="Mindctrl",
            sw_version=self.hass.data[DOMAIN][self._entry.entry_id].get(
                DATA_MINDCTRL_VERSION
            ),
            configuration_url=config_url,
        )
