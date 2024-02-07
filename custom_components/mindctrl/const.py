import logging

DOMAIN = "mindctrl"

ADDON_NAME = "mindctrl"
# https://developers.home-assistant.io/docs/add-ons/communication#network
# https://github.com/home-assistant/supervisor/blob/4ac7f7dcf08abb6ae5a018536e57d078ace046c8/supervisor/store/utils.py#L17
# ADDON_SLUG = "b6b7edbb_mindctrl"
# ...but that doesn't work so we'll use the slug from the UI??????
ADDON_SLUG = "6692a410_mindctrl"
ADDON_HOST = ADDON_SLUG.replace("_", "-")
ADDON_PORT = 5002


_LOGGER = logging.getLogger(DOMAIN)

CONF_USE_ADDON = "use_addon"
CONF_URL = "url"
CONF_INTEGRATION_CREATED_ADDON = "integration_created_addon"
CONF_ADDON_LOG_LEVEL = "addon_log_level"


SERVICE_INVOKE_MODEL = "invoke_model"
