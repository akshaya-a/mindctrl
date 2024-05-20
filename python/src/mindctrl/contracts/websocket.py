from typing import Union

from pydantic import BaseModel, Field


class BaseMindctrlMessage(BaseModel):
    id: int
    type: str


# This is the UI contract, drop tool calling
class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str


class UserMessage(BaseModel):
    role: str = "user"
    content: str


class ChatMessage(BaseMindctrlMessage):
    type: str = "mindctrl.chat"
    message: Union[AssistantMessage, UserMessage] = Field(discriminator="role")


class SubscribeMessage(BaseMindctrlMessage):
    type: str = "mindctrl.subscribe"
    subscription: str


class WebSocketMessage(BaseMindctrlMessage):
    message: Union[ChatMessage, SubscribeMessage]
