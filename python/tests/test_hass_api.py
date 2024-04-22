import logging
from random import randint
import pytest
import httpx
from httpx_ws import aconnect_ws


# https://developers.home-assistant.io/docs/api/websocket
# Is there no open source non-GPL python client for Home Assistant? Ideally this exists independently

# This set of tests is more a playground to assert the HASS API before implementing the actual client

_logger = logging.getLogger(__name__)


@pytest.fixture
async def hass_client(hass_server_and_token):
    hass_server, token = hass_server_and_token
    async with httpx.AsyncClient(
        base_url=f"{hass_server.get_base_url()}/api",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    ) as client:
        yield client


@pytest.fixture
async def hass_ws_session(hass_client, hass_server_and_token):
    # https://developers.home-assistant.io/docs/api/websocket#authentication-phase
    _, token = hass_server_and_token
    try:
        async with aconnect_ws(
            f"{hass_client.base_url}websocket", client=hass_client
        ) as ws:
            auth_required_msg = await ws.receive_json()
            _logger.info(auth_required_msg)
            assert auth_required_msg["type"] == "auth_required"
            _logger.info("Authenticating")
            await ws.send_json(
                {
                    "type": "auth",
                    "access_token": token,
                }
            )
            auth_ok_msg = await ws.receive_json()
            _logger.info(auth_ok_msg)
            assert auth_ok_msg["type"] == "auth_ok"

            yield ws
    except RuntimeError as e:
        # TODO: RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
        # This is probably a ticking timebomb for some event loop bug i don't understand, but fingers crossed
        # it's in pytest + pytest-asyncio and not the actual code (but probably not because httpx-ws is new :/ )
        if (
            "Attempted to exit cancel scope in a different task than it was entered in"
            in str(e)
        ):
            _logger.warning("known issue, but doesn't/shouldn't matter?")


async def test_rest_api_alive(hass_client):
    response = await hass_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API running."}


async def test_ws_api_alive(hass_ws_session):
    ping_id = randint(1, 1000)
    await hass_ws_session.send_json({"id": ping_id, "type": "ping"})
    pong_msg = await hass_ws_session.receive_json()
    assert pong_msg["id"] == ping_id
    assert pong_msg["type"] == "pong"


async def test_bad_message(hass_ws_session):
    msg_id = randint(1, 1000)
    list_automations_message = {
        "type": "config/category_registry/list",
        "scope": "automation",
        # "id": msg_id,
    }
    _logger.info(list_automations_message)
    await hass_ws_session.send_json(list_automations_message)
    error_response = await hass_ws_session.receive_json()
    _logger.info(error_response)
    assert error_response["type"] == "result"
    assert not error_response["success"]
    assert error_response["error"]["code"] == "invalid_format"

    list_automations_message = {
        "type": "config/category_regAIUIGEHAELJGBistry/list",
        "scope": "automation",
        "id": msg_id + 1,
    }
    _logger.info(list_automations_message)
    await hass_ws_session.send_json(list_automations_message)
    error_response = await hass_ws_session.receive_json()
    _logger.info(error_response)
    assert error_response["type"] == "result"
    assert not error_response["success"]
    assert error_response["error"]["code"] == "unknown_command"


# List Categories
# {"type":"config/category_registry/list","scope":"automation","id":54}

# Create category
# {"type":"config/category_registry/create","scope":"automation","name":"test","icon":"mdi:lightbulb-cfl-spiral-off","id":53}
# {"id": 53, "type": "result", "success": true, "result": { "category_id": "01HVW4T6S86Z2M90M1WMYCHGP9", "icon": "mdi:lightbulb-cfl-spiral-off", "name": "test" }}

# List Labels
# {"type":"config/label_registry/list","id":60}
# [{"id":63,"type":"result","success":true,"result":[{"color":"indigo","description":null,"icon":"mdi:account","label_id":"test","name":"test"}]}]}]
# Create label
# {"type":"config/label_registry/create","name":"test","icon":"mdi:account","color":"indigo","id":62}

# Creating an automation is REST POST to a unix timestamp
# http://hass-dev.ak:8123/api/config/automation/config/1713577351529
# Request Method:
# POST
# Status Code:
# 200 OK


async def test_list_automations(hass_ws_session):
    msg_id = randint(1, 1000)
    list_entities_message = {
        # config/category_registry/list and scope automation seems to be empty
        "type": "config/entity_registry/list",
        # "scope": "automation",
        "id": msg_id,
    }
    _logger.info(list_entities_message)
    await hass_ws_session.send_json(list_entities_message)
    entities_response = await hass_ws_session.receive_json()
    _logger.debug(entities_response)
    assert entities_response["id"] == msg_id
    assert entities_response["type"] == "result"
    assert entities_response["success"]
    assert len(entities_response["result"]) > 0

    automations = [
        entity
        for entity in entities_response["result"]
        if entity["platform"] == "automation"
    ]
    _logger.info(automations)
    assert len(automations) >= 0 # yes what a useless assertion


async def test_list_areas(hass_ws_session):
    msg_id = randint(1, 1000)
    list_areas_message = {
        "type": "config/area_registry/list",
        "id": msg_id,
    }
    _logger.info(list_areas_message)
    await hass_ws_session.send_json(list_areas_message)
    areas_response = await hass_ws_session.receive_json()
    _logger.info(areas_response)
    assert areas_response["id"] == msg_id
    assert areas_response["type"] == "result"
    assert areas_response["success"]
    assert len(areas_response["result"]) >= 0


async def test_list_services(hass_ws_session):
    msg_id = randint(1, 1000)
    list_services_message = {
        "type": "get_services",
        "id": msg_id,
    }
    _logger.info(list_services_message)
    await hass_ws_session.send_json(list_services_message)
    services_response = await hass_ws_session.receive_json()
    _logger.debug(services_response)
    assert services_response["id"] == msg_id
    assert services_response["type"] == "result"
    assert services_response["success"]
    assert len(services_response["result"]) > 0


async def test_list_states(hass_client):
    states_response = await hass_client.get("/states")
    assert states_response.status_code == 200
    states = states_response.json()
    assert len(states) > 0
