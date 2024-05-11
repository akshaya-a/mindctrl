import logging

from pydantic import BaseModel

from dapr.clients import DaprClient
from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
)

from mindctrl.mlmodels import invoke_model_impl, ModelInvocation

_logger = logging.getLogger(__name__)


# TODO: unify with the monkeypatch of deploymentserver for tool calling?
class Message(BaseModel):
    content: str
    role: str


class Conversation(BaseModel):
    messages: list[Message]


def append_message(ctx: WorkflowActivityContext, message_json: str):
    _logger.info(f"Received message: {message_json}")
    # Moving pydantic away from the signature because it's not json.dumps-able
    # TODO: file a bug on durabletask.internal.shared.py
    message = Message.model_validate_json(message_json)
    # This is where we do some fun memory tricks like compression, embedding, windowing etc
    # TODO: Dapr + async def activities?
    with DaprClient() as d:
        store_name = "daprstore"
        convo_id = f"convo-{ctx.workflow_id}"
        current_convo = d.get_state(store_name=store_name, key=convo_id)
        # TODO: Handle etags
        convo = Conversation(messages=[])
        if current_convo.data:
            convo = Conversation.model_validate_json(current_convo.data)
        convo.messages.append(message)
        conversation_length = len(convo.messages)
        d.save_state(store_name, convo_id, convo.model_dump_json())
    return conversation_length


def invoke_model(
    ctx: WorkflowActivityContext, input: ModelInvocation
) -> dict[str, str]:
    # TODO: Handle more complex responses
    _logger.info(f"Invoking model with input: {input}")
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(input.model_uri)
    response_message = invoke_model_impl(
        model, input.payload, input.scenario_name, input.input_variables
    )[0]
    return {"content": response_message, "role": "assistant"}


def conversation_turn_workflow(ctx: DaprWorkflowContext, input: ModelInvocation):
    # This workflow handles a conversation between a user and LLM. It will receive a message from the user, and then
    # respond with a message from LLM. This will continue until the user sends a message that ends the conversation.
    # The conversation will be stored in the state store, so that it can be resumed later.

    _logger.info(f"{input}")

    # 1. Activity: Append message to conversation history, return conversation position
    # 2. Activity: Input conversation position, Fetch conversation history, Invoke model, return response
    # 3. Activity: Append response to conversation history, return conversation position

    # TODO: this is wrong, get a real message structure for chat models
    message = Message(content=str(input.payload), role="user")
    conversation_status: int = yield ctx.call_activity(
        append_message, input=message.model_dump_json()
    )
    _logger.info(f"Conversation turn: {conversation_status}")
    response: dict[str, str] = yield ctx.call_activity(invoke_model, input=input)
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

    response_message = Message(**response)
    conversation_status = yield ctx.call_activity(
        append_message, input=response_message.model_dump_json()
    )
