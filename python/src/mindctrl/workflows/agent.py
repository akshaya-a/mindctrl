import logging
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import mlflow.pyfunc
from dapr.clients import DaprClient
from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
)

from mindctrl.mlmodels import invoke_model_impl
from mindctrl.openai_deployment import _ContentFormatter, _OpenAIDeploymentWrapper

_logger = logging.getLogger(__name__)


# TODO: unify with the monkeypatch of deploymentserver for tool calling?
# TODO: Fix durable task client to handle pydantic models, then switch
# TODO: file a bug on durabletask.internal.shared.py
@dataclass
class Message:
    content: str
    role: str


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
            "columns": ["query"],
            "data": [[query]],
        }
    }


def append_history(model: mlflow.pyfunc.PyFuncModel, history: Conversation):
    inner_model = model._model_impl
    assert isinstance(inner_model, _OpenAIDeploymentWrapper)
    messages = [m.__dict__ for m in history.messages]
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
        payload = get_user_chat_payload(history.messages[-1].content)

        del history.messages[-1]
        model = append_history(model, history)

        response_message = invoke_model_impl(
            model, payload, input.scenario_name, input.input_variables
        )[0]
        return Message(content=response_message, role="assistant")
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        # breakpoint()
        raise


def conversation_turn_workflow(ctx: DaprWorkflowContext, input: ModelInvocation):
    _logger.info(f"{input}")

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
        # if tool call
        # messages.append(response.choices[0].message)
        # then yield ctx.call_activity(invoke_tool, input=response.choices[0].tool)
        # messages.append(
        #     {
        #         "role": "function",
        #         "name": function_name,
        #         "content": str(function_response),
        #     }
        # )
        # messages

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
