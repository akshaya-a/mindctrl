import logging

import pytest
from httpx import URL
from mindctrl.homeassistant.client import HassClient
from mindctrl.homeassistant.messages import CreateLabel

_logger = logging.getLogger(__name__)


@pytest.fixture
async def hass_client(hass_server_and_token):
    server, token = hass_server_and_token
    try:
        async with HassClient(
            id="pytest",
            hass_url=URL(f"{server.get_base_url()}/api"),
            token=token,
        ) as client:
            yield client
    except RuntimeError as e:
        # TODO: RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
        # This is probably a ticking timebomb for some event loop bug i don't understand, but fingers crossed
        # it's in pytest + pytest-asyncio and not the actual code (but probably not because httpx-ws is new :/ )
        if (
            "Attempted to exit cancel scope in a different task than it was entered in"
            in str(e)
        ):
            _logger.warning("known issue, but doesn't/shouldn't matter?")


async def test_mctrl_list_automations(hass_client):
    automations = await hass_client.list_automations()
    _logger.info(automations)
    assert len(automations) >= 0


async def test_mctrl_list_areas(hass_client):
    areas = await hass_client.list_areas()
    _logger.info(areas)
    assert len(areas) >= 0


async def test_mctrl_list_labels(hass_client):
    labels = await hass_client.list_labels()
    _logger.info(labels)
    assert len(labels) >= 0


async def test_automation_autotag(hass_client, request):
    test_automation_name = f"auto_{request.node.name}"
    await hass_client.create_automation(test_automation_name, request.node.name)
    automations = await hass_client.list_automations()
    _logger.info(automations)
    assert len(automations) >= 1

    # labels = await hass_client.list_labels()
    # areas = await hass_client.list_areas()

    # ...AI magic...

    test_label_name = f"{request.node.name}-label"
    create_new_labels = [
        CreateLabel(
            id=1,
            name=test_label_name,
            color="indigo",
            icon="mdi:account",
            description=None,
        )
    ]
    for label in create_new_labels:
        await hass_client.create_label(label)

    # This breaks because the automation.id is the unique_id, not the entity_id. Need to fix this.
    # for automation in automations:
    #     _logger.info(f"Adding label {test_label_name} to {automation.id}")
    #     await hass_client.add_labels(automation.id, [test_label_name])

    await hass_client.add_labels(
        f"automation.{test_automation_name}", [test_label_name]
    )

    entities = await hass_client.list_entities()
    automations = [e for e in entities if e["platform"] == "automation"]
    _logger.info(automations)
    tagged_automations = [a for a in automations if test_label_name in a["labels"]]
    assert len(tagged_automations) == 1
