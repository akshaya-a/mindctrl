import asyncio
import logging
import time
from typing import Union

import httpx
from httpx_ws import aconnect_ws
from pydantic import ValidationError

from .messages import (
    AreasResult,
    Auth,
    AuthChallenge,
    AuthOk,
    Automation,
    Command,
    CreateAutomation,
    CreateLabel,
    Error,
    Label,
    LabelsResult,
    ListAreas,
    ListEntities,
    ListLabels,
    ManyResponsesWrapper,
    Result,
    SingleResponseWrapper,
    UpdateEntityLabels,
)

_logger = logging.getLogger(__name__)


class HassClientError(Exception):
    pass


class HassClient(object):
    def __init__(self, id: str, hass_url: httpx.URL, token: str):
        self.hass_url = hass_url
        self._token = token
        if not (str(self.hass_url).endswith("api")):
            raise ValueError(
                f"hass_url must end with 'api'. For example, 'http://homeassistant.local:8123/api'. Got: {self.hass_url}"
            )
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.websocket_message_id = 0
        self._client = httpx.AsyncClient(base_url=self.hass_url, headers=self.headers)
        self._ws_session = aconnect_ws(
            f"{self._client.base_url}websocket", client=self._client
        )
        self._authenticated_session = None

        # is there a better way to process these?
        self._command_results: dict[int, Union[Result, Error]] = {}

    @property
    def authenticated_session(self):
        if not self._authenticated_session:
            raise ValueError("Session not authenticated")
        return self._authenticated_session

    async def __aenter__(self):
        session = await self._ws_session.__aenter__()

        auth_required_msg = AuthChallenge.model_validate(await session.receive_json())
        _logger.info(auth_required_msg)

        await session.send_json(Auth(access_token=self._token).model_dump())

        auth_ok_msg = AuthOk.model_validate(await session.receive_json())
        _logger.info(auth_ok_msg)

        self._authenticated_session = session
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if len(self._command_results.keys()) > 0:
            _logger.warning(
                f"Exiting with {len(self._command_results.keys())} unprocessed messages"
            )
        await self._ws_session.__aexit__(exc_type, exc, tb)
        await self._client.aclose()

    async def _receive_message(self, message_id: int) -> Union[Result, None]:
        if message_id in self._command_results.keys():
            val = self._command_results.pop(message_id)
            if not val.success:
                assert isinstance(val, Error)
                raise HassClientError(f"Error: {val.code} - {val.message}")
            assert isinstance(val, Result)
            return val

        response_json = await self.authenticated_session.receive_json()
        responses: list[Union[Result, Error]]
        try:
            response = SingleResponseWrapper.model_validate(
                {"response": response_json}, strict=False
            )
            responses = [response.response]
        except ValidationError as ve:
            _logger.info(f"receive: {ve}")
            response = ManyResponsesWrapper.model_validate(
                {"responses": response_json}, strict=False
            )
            responses = response.responses

        return_response: Union[Result, Error, None] = None
        for response in responses:
            if response.id != message_id:
                self._command_results[response.id] = response
            else:
                return_response = response

        if isinstance(return_response, Error):
            raise HassClientError(
                f"Error: {return_response.code} - {return_response.message}"
            )

        return return_response

    async def _send_message(self, message: Command) -> Result:
        # The atomicity of this is sketchy - add tests
        self.websocket_message_id += 1
        message.id = self.websocket_message_id
        await self.authenticated_session.send_json(message.model_dump())
        result = None
        while result is None:
            result = await self._receive_message(message.id)
            if result is None:
                _logger.info("Recent receive batch didn't have the response")
                await asyncio.sleep(0.1)
        return result

    @staticmethod
    def _current_milli_time():
        return round(time.time() * 1000)

    async def list_entities(self):
        entities = await self._send_message(ListEntities(id=-1))
        if entities.result is None:
            _logger.warning(f"Unexpected null entities result: {entities}")
            return []
        return entities.result

    # TODO: bad api call pattern, need to revisit
    async def list_automations(self) -> list[Automation]:
        entities = await self.list_entities()
        _logger.debug(entities)
        automation_entities = [
            entity for entity in entities if entity["platform"] == "automation"
        ]

        _logger.info(f"Fetching {len(automation_entities)} automations")
        fetch_automation_tasks = []
        for entity in automation_entities:
            fetch_automation_tasks.append(self.get_automation(entity["unique_id"]))
        return await asyncio.gather(*fetch_automation_tasks)

    async def list_labels(self):
        any_result = await self._send_message(ListLabels(id=-1))
        labels = LabelsResult.model_validate_json(any_result.model_dump_json())
        return labels.result

    async def list_areas(self):
        any_result = await self._send_message(ListAreas(id=-1))
        areas = AreasResult.model_validate_json(any_result.model_dump_json())
        return areas.result

    async def create_label(self, label: Label):
        await self._send_message(
            CreateLabel(
                color=label.color,
                description=label.description,
                icon=label.icon,
                name=label.name,
                id=-1,
            )
        )
        return

    async def add_labels(self, entity_id: str, labels: list[str]):
        await self._send_message(
            UpdateEntityLabels(
                entity_id=entity_id,
                labels=labels,
                id=-1,
            )
        )

    async def get_automation(self, unique_id: str):
        # Creating an automation is REST POST to a unix timestamp
        # http://hass-dev.ak:8123/api/config/automation/config/1713577351529
        # Request Method:
        # POST
        # Status Code:
        # 200 OK
        path = f"config/automation/config/{unique_id}"
        get_response = await self._client.get(path)
        get_response.raise_for_status()
        return Automation.model_validate(get_response.json())

    async def create_automation(self, name: str, description: str):
        # Creating an automation is REST POST to a unix timestamp
        # http://hass-dev.ak:8123/api/config/automation/config/1713577351529
        # Request Method:
        # POST
        # Status Code:
        # 200 OK
        current_milli_time = HassClient._current_milli_time()
        path = f"config/automation/config/{current_milli_time}"
        response = await self._client.post(
            path,
            json=CreateAutomation(
                alias=name,
                description=description,
                mode="single",
                action=[],
                condition=[],
                trigger=[],
            ).model_dump(),
        )
        response.raise_for_status()
        # response.json() is just {'result': 'ok'}, need to do a get (why?)
        return await self.get_automation(str(current_milli_time))
