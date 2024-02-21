import logging
import asyncio
from typing import Tuple

import mlflow
import openai
from mlflow.entities.model_registry import RegisteredModel
from mlflow import MlflowClient


_logger = logging.getLogger(__name__)

TIMERANGE_MODEL = "timerange"
CHAT_MODEL = "chat"
SUMMARIZER_MODEL = "summarizer"
EMBEDDINGS_MODEL = "embeddings"

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

OAI_GPT4_T = "gpt-4-turbo-preview"
SUMMARIZER_OAI_MODEL = OAI_GPT4_T
CHAT_OAI_MODEL = OAI_GPT4_T

SUMMARIZATION_PROMPT = """You're an AI assistant for home automation. You're being given the latest set of events from the home automation system. You are to concisely summarize the events relevant to the user's query followed by an explanation of your reasoning.
    EXAMPLE SENSOR DATA:
    {{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.washer_wash_completed', 'old_state': {{'entity_id': 'binary_sensor.washer_wash_completed', 'state': 'off', 'attributes': {{'friendly_name': 'Washer Wash completed'}}, 'last_changed': '2023-12-23T09:20:07.695950+00:00', 'last_updated': '2023-12-23T09:20:07.695950+00:00', 'context': {{'id': '01HJAZK20FGR2Z9NTCD46XMQEG', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.washer_wash_completed', 'state': 'on', 'attributes': {{'friendly_name': 'Washer Wash completed'}}, 'last_changed': '2023-12-23T09:53:07.724686+00:00', 'last_updated': '2023-12-23T09:53:07.724686+00:00', 'context': {{'id': '01HJB1FFMCT64MM6GKQX55HMKQ', 'parent_id': None, 'user_id': None}}}}}}}}
    {{'event_type': 'call_service', 'event_data': {{'domain': 'tts', 'service': 'cloud_say', 'service_data': {{'cache': True, 'entity_id': ['media_player.kitchen_interrupt', 'media_player.master_bedroom_interrupt'], 'message': 'The washer is complete! Move the clothes to the dryer or they gonna get so so so stinky poo poo!!!!'}}}}}}
    {{'event_type': 'call_service', 'event_data': {{'domain': 'media_player', 'service': 'play_media', 'service_data': {{'entity_id': ['media_player.kitchen_interrupt', 'media_player.master_bedroom_interrupt'], 'media_content_id': 'media-source://tts/cloud?message=The+washer+is+complete!+Move+the+clothes+to+the+dryer+or+they+gonna+get+so+so+so+stinky+poo+poo!!!!&cache=true', 'media_content_type': 'music', 'announce': True}}}}}}
    {{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'old_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'off', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:46:48.945634+00:00', 'last_updated': '2023-12-23T09:46:48.945634+00:00', 'context': {{'id': '01HJB13XQHGYJYCBH1BS9E6JQY', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'on', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:26.786268+00:00', 'last_updated': '2023-12-23T09:53:26.786268+00:00', 'context': {{'id': '01HJB1G282MSCK7H5KDVE5S260', 'parent_id': None, 'user_id': None}}}}}}}}
    {{'event_type': 'state_changed', 'event_data': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'old_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'on', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:26.786268+00:00', 'last_updated': '2023-12-23T09:53:26.786268+00:00', 'context': {{'id': '01HJB1G282MSCK7H5KDVE5S260', 'parent_id': None, 'user_id': None}}}}, 'new_state': {{'entity_id': 'binary_sensor.bedroom_motion_sensor_motion', 'state': 'off', 'attributes': {{'device_class': 'motion', 'friendly_name': 'Bedroom motion sensor Motion'}}, 'last_changed': '2023-12-23T09:53:50.016556+00:00', 'last_updated': '2023-12-23T09:53:50.016556+00:00', 'context': {{'id': '01HJB1GRY0RSE049YV3NPJ6QFC', 'parent_id': None, 'user_id': None}}}}}}}}

    EXAMPLE QUERY: "Is the laundry running?"
    EXAMPLE OUTPUT: "The washer is complete. You should move the clothes to the dryer. I see the washer completed sensor turned on at 2023-12-23T09:20:07.695950+00:00"

    EXAMPLE QUERY: "Is there anyone in the bedroom?"
    EXAMPLE OUTPUT: "There is no one in the bedroom. Even though there was recent activity, I see the bedroom motion sensor turned off at 2023-12-23T09:53:50.016556+00:00"

    EXAMPLE QUERY: "What rooms have activity recently?"
    EXAMPLE OUTPUT: "The bedroom has had activity. I see the bedroom motion sensor turned on at 2023-12-23T09:53:26.786268+00:00"

    Remember to be concise and that there could be multiple sequences of events interleaved, so you can output multiple lines.
    """


def set_alias(client: MlflowClient, model_name: str, alias: str):
    filter_string = f"name='{model_name}'"
    latest_version = client.search_model_versions(
        filter_string, max_results=1, order_by=["name DESC"]
    )[0]
    print(f"Setting alias {alias} for {model_name} version {latest_version.version}")
    client.set_registered_model_alias(model_name, alias, latest_version.version)


def log_system_models(force_publish=False) -> list[RegisteredModel]:
    mlflow_client = MlflowClient()

    rms = [rm for rm in mlflow_client.search_registered_models()]
    registry_models: list[str] = [rm.name for rm in rms]
    print(f"Already registered models: {registry_models}")

    QUERY_PROMPT = """
    You are given a query about events or actions happening for home automation over an unknown period of time.
    Based on the input user's question, return the start and end times for the query. Prefix lines with THOUGHT: to decide next steps. Show your work.

    EXAMPLE INPUT: How many people are in the house?
    EXAMPLE THOUGHT: I need to detect recent motion, so an hour lookback for motion sensors should be sufficient.
    EXAMPLE THOUGHT: The current time is 2024-01-02 22:27:46.379954, which is the end time. Therefore the start time is 2024-01-02 21:27:46.379954
    EXAMPLE OUTPUT: {{"start": "2024-01-02 21:27:46.379954", "end": "2024-01-02 22:27:46.379954"}}

    OUTPUT ONLY JSON FORMATTED OBJECTS WITH "start" and "end" keys!
    """
    if TIMERANGE_MODEL not in registry_models or force_publish:
        mlflow.openai.log_model(
            model="gpt-3.5-turbo-0125",
            task=openai.ChatCompletion,
            messages=[
                {"role": "system", "content": QUERY_PROMPT},
                {"role": "user", "content": "INPUT: {query}"},
            ],
            artifact_path="oai-timerange",
            registered_model_name=TIMERANGE_MODEL,
        )
        set_alias(mlflow_client, TIMERANGE_MODEL, CHAMPION_ALIAS)

    if CHAT_MODEL not in registry_models or force_publish:
        mlflow.openai.log_model(
            model=CHAT_OAI_MODEL,
            task=openai.ChatCompletion,
            messages=[
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {
                    "role": "user",
                    "content": "SENSOR DATA:\n{state_lines}\n\nQUERY: {query}",
                },
            ],
            artifact_path="oai-chat",
            registered_model_name=CHAT_MODEL,
        )
        set_alias(mlflow_client, CHAT_MODEL, CHAMPION_ALIAS)

    if SUMMARIZER_MODEL not in registry_models or force_publish:
        mlflow.openai.log_model(
            model=SUMMARIZER_OAI_MODEL,
            task=openai.ChatCompletion,
            messages=[
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {
                    "role": "user",
                    "content": """
------START SENSOR DATA-----:\n
{state_lines}\n
------END SENSOR DATA-----\n
QUERY: summarize the above events for me""",
                },
            ],
            artifact_path="oai-summarizer",
            registered_model_name=SUMMARIZER_MODEL,
        )
        set_alias(mlflow_client, SUMMARIZER_MODEL, CHAMPION_ALIAS)

    if EMBEDDINGS_MODEL not in registry_models or force_publish:
        mlflow.openai.log_model(
            model="text-embedding-ada-002",
            task=openai.Embedding,
            artifact_path="oai-embeddings",
            registered_model_name=EMBEDDINGS_MODEL,
        )
        set_alias(mlflow_client, EMBEDDINGS_MODEL, CHAMPION_ALIAS)

        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        data = "input data for signature inference"
        signature = mlflow.models.infer_signature(
            model_input=data,
            model_output=embedding_model.encode(data),
        )

        mlflow.sentence_transformers.log_model(
            model=embedding_model,
            artifact_path="st-embeddings",
            registered_model_name=EMBEDDINGS_MODEL,
            signature=signature,
            input_example=data,
        )
        set_alias(mlflow_client, EMBEDDINGS_MODEL, CHALLENGER_ALIAS)

    return rms


# TODO: Add webhooks/eventing to MLflow OSS server. AzureML has eventgrid support
# In its absence, we poll the MLflow server for changes to the model registry
async def poll_registry(delay_seconds: float = 10.0):
    while True:
        # Sync any new models by tag/label/all
        # Solve any environment dependencies mismatch or fail
        # TODO: Consider running a separate server for each model to solve the isolation problem
        _logger.debug("Polling registry for changes")
        await asyncio.sleep(delay=delay_seconds)


def summarize_events(
    events: list[str], include_challenger=False
) -> Tuple[list[str], list[str]]:
    champion_model = mlflow.pyfunc.load_model(f"models:/summarizer@{CHAMPION_ALIAS}")
    champion_summary = [""]
    try:
        champion_summary = champion_model.predict(events)
    except Exception as e:  # noqa: E722
        _logger.warning(f"Failed to load champion model: {e}")
        pass

    challenger_summary = [""]
    if include_challenger:
        try:
            challenger_model = mlflow.pyfunc.load_model(
                f"models:/summarizer@{CHALLENGER_ALIAS}"
            )
            challenger_summary = challenger_model.predict(events)
        except Exception as e:  # noqa: E722
            _logger.warning(f"Failed to load challenger model: {e}")
            pass

    return champion_summary, challenger_summary


def tokenized_events(events: list[str]) -> list[str]:
    model = mlflow.sentence_transformers.load_model("models:/localembeddings/latest")
    return model.predict(events)


def embed_summary(summary: str) -> list[float]:
    # TODO: switch to pyfunc after you test bare flavor
    model = mlflow.sentence_transformers.load_model("models:/localembeddings/latest")
    # return model.predict(summary)
    return model.encode(summary).tolist()
