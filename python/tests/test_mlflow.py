import json
import mlflow
import mlflow.openai
import openai
import uuid

from mindctrl.const import SCENARIO_NAME_PARAM
from mindctrl.mlmodels import log_system_models
from mindctrl.openai_deployment import log_model


def test_mlflow_setup(mlflow_fluent_session):
    assert "sqlite" in mlflow.get_tracking_uri()


def test_log_system_models(mlflow_fluent_session):
    log_system_models()

    rms = mlflow.search_registered_models()
    assert len(rms) == 4
    assert "summarizer" in [rm.name for rm in rms]


def test_log_simple_chat_completions(mlflow_fluent_session):
    rm_name = uuid.uuid4().hex
    log_model(
        model="gpt-3.5-turbo-0125",
        task=openai.chat.completions,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "{question}"},
        ],
        artifact_path=rm_name,
        registered_model_name=rm_name,
    )

    mvs = mlflow.search_model_versions(filter_string=f"name = '{rm_name}'")
    assert len(mvs) == 1


def test_log_tool_calling(mlflow_fluent_session, request):
    rm_name = uuid.uuid4().hex
    log_model(
        model="gpt-3.5-turbo-0125",
        task=openai.chat.completions,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "{question}"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="auto",
        artifact_path=rm_name,
        registered_model_name=rm_name,
    )

    mvs = mlflow.search_model_versions(filter_string=f"name = '{rm_name}'")
    assert len(mvs) == 1

    model = mlflow.pyfunc.load_model(mvs[0].source)
    result = model.predict(
        {"question": "What's the weather in San Francisco?"},
        params={SCENARIO_NAME_PARAM: request.node.name},
    )
    assert len(result) == 1
    tool_calls = result[0]
    assert len(tool_calls) == 1
    function_call = tool_calls[0]
    assert function_call["type"] == "function"
    assert function_call["function"]["name"] == "get_current_weather"
    arguments = json.loads(function_call["function"]["arguments"])
    assert arguments["location"] == "San Francisco"
