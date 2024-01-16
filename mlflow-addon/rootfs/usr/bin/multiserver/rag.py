from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from fastapi import Request
import json
from pydantic import BaseModel


class EventType(Enum):
    state_changed = 1
    call_service = 2


class Event(BaseModel):
    event_type: EventType
    event_data: object


class StateRetriever(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieve_events(self, start: datetime, end: datetime) -> list[Event]:
        pass


class RingBufferRetriever(StateRetriever):
    def __init__(self, request: Request) -> None:
        self.buffer = request.state.state_ring_buffer
        super().__init__()

    def retrieve_events(self, start: datetime, end: datetime) -> list[Event]:
        return [Event(event_type=EventType.call_service, event_data={})]


def retrieve_events(request: Request, start: datetime, end: datetime) -> list[Event]:
    return RingBufferRetriever(request).retrieve_events(start, end)


def extract_timestamps(query_range_response: str) -> tuple[datetime, datetime]:
    import re

    return_parser = r"(\{.*\})"
    match = re.match(return_parser, query_range_response)
    if not match or not match.lastgroup:
        raise Exception(f"LLM returned malformed output: {query_range_response}")

    obj = json.loads(match.lastgroup)
    return obj["start"], obj["end"]
