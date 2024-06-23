import json
import logging
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Generic, Optional, TypeVar

import mlflow.pyfunc
from dapr.clients import DaprClient
from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
)
from pydantic import BaseModel

from mindctrl.mlmodels import invoke_model_impl
from mindctrl.openai_deployment import _ContentFormatter, _OpenAIDeploymentWrapper

_logger = logging.getLogger(__name__)


# TODO: THIS IS SOOOOOO BADDDDD
# agent <-> openai_deployment contract
# openai_deployment <-> replay server contract
# replay server <-> openai contract
# And all are slightly different :(
@dataclass
class Function:
    name: str
    arguments: str


@dataclass
class FunctionCall:
    id: str
    function: Function
    type: str = "function"


# TODO: unify with the monkeypatch of deploymentserver for tool calling?
# TODO: Fix durable task client to handle pydantic models, then switch
# TODO: file a bug on durabletask.internal.shared.py
@dataclass
class Message:
    content: Optional[str]
    role: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[FunctionCall]] = None


@dataclass
class Conversation:
    messages: list[Message]


@dataclass
class MessageInConvo:
    message: Message
    conversation_id: str


@dataclass
class ModelInvocation:
    model_uri: str
    scenario_name: Optional[str]
    input_variables: dict[str, str]
    conversation_id: str
    history: Optional[Conversation]


def append_message(
    ctx: WorkflowActivityContext, message: MessageInConvo
) -> Conversation:
    _logger.info(f"Received message: {message}")
    try:
        # This is where we do some fun memory tricks like compression, embedding, windowing etc
        # TODO: Dapr + async def activities?
        with DaprClient() as d:
            store_name = "daprstore"
            convo_id = f"convo-{message.conversation_id}"
            current_convo = d.get_state(store_name=store_name, key=convo_id)

            # TODO: Handle etags
            convo = Conversation(messages=[])
            if current_convo.data:
                convo: Conversation = pickle.loads(current_convo.data)

            # The type marshaling by durabletask/dapr is a bit wonky...
            if isinstance(message.message, dict):
                # if
                convo.messages.append(Message(**message.message))
            elif isinstance(message.message, Message):
                convo.messages.append(message.message)
            elif isinstance(message.message, SimpleNamespace):
                convo.messages.append(Message(**message.message.__dict__))
            else:
                raise ValueError(f"Unknown message type: {type(message.message)}")
            d.save_state(store_name, convo_id, pickle.dumps(convo))

        # TODO: Put the compressed state into the system message next time
        return convo
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        # breakpoint()
        raise


def get_user_chat_payload(query: str) -> dict:
    return {
        "dataframe_split": {
            "columns": ["content", "role", "tool_call_id", "name"],
            "data": [[query, "user", None, None]],
        }
    }


def get_tool_call_payload(tool_call_id: str, content: str, name: str) -> dict:
    return {
        "dataframe_split": {
            "columns": ["content", "role", "tool_call_id", "name"],
            "data": [[content, "tool", tool_call_id, name]],
        }
    }


def append_history(model: mlflow.pyfunc.PyFuncModel, history: Conversation):
    inner_model = model._model_impl
    assert isinstance(inner_model, _OpenAIDeploymentWrapper)
    messages = [m.__dict__ for m in history.messages]
    print("MESSAGES", messages)
    print("EXISTING MESSAGES", inner_model.template)
    existing_messages: list = inner_model.template  # type: ignore
    combined_messages = existing_messages[:1] + messages + existing_messages[1:]
    inner_model.formater = _ContentFormatter(inner_model.task, combined_messages)
    return model


# TODO: CONVERT THIS TO MESSAGE IN, MESSAGE OUT
def invoke_model(ctx: WorkflowActivityContext, input: ModelInvocation) -> Message:
    # TODO: Handle more complex responses
    print(f"Invoking model with input: {input}")
    try:
        model = mlflow.pyfunc.load_model(input.model_uri)
        assert input.history is not None
        assert len(input.history.messages) > 0
        history = Conversation(messages=[Message(**m) for m in input.history.messages])  # type: ignore
        current_message = history.messages[-1]
        if current_message.tool_call_id is not None:
            assert (
                current_message.content is not None
            ), f"Content is None: {current_message}"
            assert current_message.name is not None, f"Name is None: {current_message}"
            payload = get_tool_call_payload(
                current_message.tool_call_id,
                current_message.content,
                current_message.name,
            )
        else:
            assert current_message.content is not None
            payload = get_user_chat_payload(current_message.content)

        print(f"PAYLOAD: {payload}")
        del history.messages[-1]
        print(f"HISTORY: {history}")
        model = append_history(model, history)

        response = invoke_model_impl(
            model, payload, input.scenario_name, input.input_variables
        )
        print(response)
        response_message = response[0]
        print(response_message)
        # TODO: return a better object from openai_deployment.predict()
        if isinstance(response_message, list):
            # TODO: Support parallel tool calling
            function_call: dict = response_message[0]
            if function_call.get("type", "unknown") != "function":
                raise ValueError(f"Unknown response type: {function_call}")
            _logger.info(f"Received function call: {function_call}")
            # TODO: Why am I writing all this logic again?
            return Message(
                content=None,
                tool_calls=[
                    FunctionCall(
                        id=function_call["id"],
                        function=Function(
                            name=function_call["function"]["name"],
                            arguments=function_call["function"]["arguments"],
                        ),
                    )
                ],
                role="assistant",
            )
        return Message(content=response_message, role="assistant")
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        # breakpoint()
        raise


# TODO: Convert the common try/except pattern into a mindctrl decorator
def invoke_tool(ctx: WorkflowActivityContext, function_call: dict) -> str:
    from mindctrl.homeassistant.client import TOOL_MAP

    try:
        print(f"Invoking tool: {function_call}")
        func = TOOL_MAP.get(function_call["function"]["name"])
        if func is None:
            raise ValueError(
                f"Unknown tool: {function_call['function']['name']}, have {TOOL_MAP.keys()}"
            )
        params = json.loads(function_call["function"]["arguments"])
        _logger.info(f"Calling tool: {function_call['function']['name']} with {params}")
        tool_result = func(**params)
        _logger.info(f"Tool result: {tool_result}")
        if isinstance(tool_result, BaseModel):
            return tool_result.model_dump_json()
        return json.dumps(tool_result)
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        # breakpoint()
        raise


# def tool_turn_workflow(ctx: DaprWorkflowContext, input: ModelInvocation):
#     _logger.info(f"Calling Tool: {input}")
#     conversation: Conversation = yield ctx.call_activity(append_message, input=input)
#     tool_result = yield ctx.call_activity(invoke_tool, input=input)


def conversation_turn_workflow(ctx: DaprWorkflowContext, input: ModelInvocation):
    _logger.info(f"Starting Conversation turn: {input}")

    try:
        message_str = yield ctx.wait_for_external_event("user_message")
        assert isinstance(message_str, str)
        # TODO: this is wrong, get a real message structure for chat models
        message = Message(content=message_str, role="user")
        conversation: Conversation = yield ctx.call_activity(
            append_message,
            input=MessageInConvo(
                message=message, conversation_id=input.conversation_id
            ),
        )
        input.history = conversation
        response_message: Message = yield ctx.call_activity(invoke_model, input=input)

        tool_calling = response_message.tool_calls is not None
        while tool_calling:
            _logger.info(f"Tool calling: {response_message}")
            # TODO Make this a child workflow
            assert response_message.tool_calls is not None
            conversation: Conversation = yield ctx.call_activity(
                append_message,
                input=MessageInConvo(
                    message=response_message, conversation_id=input.conversation_id
                ),
            )
            print("CALLING TOOL", response_message)
            tool_result: str = yield ctx.call_activity(
                invoke_tool,
                input=response_message.tool_calls[0],  # type: ignore
            )
            print(f"TOOL RESULT: {tool_result}")
            tool_result_message = Message(
                role="tool",
                content=tool_result,
                tool_call_id=response_message.tool_calls[0]["id"],  # type: ignore
                name=response_message.tool_calls[0]["function"]["name"],  # type: ignore
            )
            conversation: Conversation = yield ctx.call_activity(
                append_message,
                input=MessageInConvo(
                    message=tool_result_message, conversation_id=input.conversation_id
                ),
            )
            input.history = conversation
            response_message: Message = yield ctx.call_activity(
                invoke_model, input=input
            )
            tool_calling = response_message.tool_calls is not None

            # tool_response = yield ctx.call_child_workflow(
            #     tool_turn_workflow,
            #     input=MessageInConvo(
            #         message=response_message, conversation_id=input.conversation_id
            #     ),
            # )
            # return response_message

        conversation = yield ctx.call_activity(
            append_message,
            input=MessageInConvo(
                message=response_message, conversation_id=input.conversation_id
            ),
        )
        # TODO: Need to implement https://github.com/microsoft/durabletask-python/issues/25
        # That way the response can be custom status instead of return value
        # Right now you have to schedule each turn of the workflow manually
        # ctx.continue_as_new(input)
        return response_message
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        # breakpoint()
        raise
