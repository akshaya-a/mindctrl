from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from fastapi import Request
import json
from pydantic import BaseModel
from itertools import islice


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


# https://cookbook.openai.com/examples/embedding_long_inputs
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# TODO: Need to enable this to avoid truncation for embedding
# Batched tokenization makes this a bit more interesting
# from sentence_transformers import SentenceTransformer
# def chunked_tokens(text, sentence_transformer: SentenceTransformer):
#     tokens = sentence_transformer.tokenize(text)
#     encoding = tiktoken.get_encoding(encoding_name)
#     tokens = encoding.encode(text)
#     chunks_iterator = batched(tokens, chunk_length)
#     yield from chunks_iterator
