from typing import Any, Optional, Union
from pydantic import BaseModel


class Message(BaseModel):
    type: str


class AuthChallenge(Message):
    type: str = "auth_required"
    ha_version: str


class AuthOk(Message):
    type: str = "auth_ok"
    ha_version: str


class Auth(Message):
    type: str = "auth"
    access_token: str


class Command(Message):
    id: int


class CommandResponse(Command):
    type: str = "result"
    success: bool


class Error(CommandResponse):
    success: bool = False
    code: str
    message: str


class Result(CommandResponse):
    success: bool = True
    result: Optional[Any]


class ManyResponsesWrapper(BaseModel):
    responses: list[Union[Error, Result]]


class SingleResponseWrapper(BaseModel):
    response: Union[Error, Result]


class ListEntities(Command):
    type: str = "config/entity_registry/list"


class ListLabels(Command):
    type: str = "config/label_registry/list"


class ListAreas(Command):
    type: str = "config/area_registry/list"


# {"color":"indigo","description":null,"icon":"mdi:account","label_id":"test","name":"test"}
class Label(BaseModel):
    color: str
    description: Optional[str]
    icon: str
    label_id: str
    name: str


class LabelsResult(Result):
    result: list[Label]


class Area(BaseModel):
    area_id: str
    name: str
    aliases: list[str]
    floor_id: Optional[str]
    icon: Optional[str]
    labels: list[str]
    picture: Optional[str]


class AreasResult(Result):
    result: list[Area]


# {"type":"config/label_registry/create","name":"test","icon":"mdi:account","color":"indigo","id":62}
class CreateLabel(Command):
    type: str = "config/label_registry/create"
    description: Optional[str]
    name: str
    icon: str
    color: str


# {"type":"config/entity_registry/update","entity_id":"automation.test_zone_automation","labels":["test"],"id":51}
class UpdateEntityLabels(Command):
    type: str = "config/entity_registry/update"
    entity_id: str
    labels: list[str]


class CreateAutomation(BaseModel):
    action: list[Any]
    alias: str
    condition: list[Any]
    description: str
    mode: str
    trigger: list[Any]


class Automation(CreateAutomation):
    id: str
