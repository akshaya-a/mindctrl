import collections
import logging
from fastapi import APIRouter, Request, HTTPException
import mlflow

from mindctrl.const import SCENARIO_NAME_HEADER
from mindctrl.mlmodels import (
    SUMMARIZATION_PROMPT,
    SUMMARIZER_OAI_MODEL,
    invoke_model_impl,
)

router = APIRouter(prefix="/deployed-models", tags=["deployed_models"])


_logger = logging.getLogger(__name__)


# from .rag import extract_timestamps, retrieve_events


# # LLM decide query timerange -> statelookup -> in-mem faiss -> query index
# # OAI functioncalling/guided prompt -> llamaindex "docstore" -> lli
# def invocation_pipeline(request: Request, query: str):
#     range_model = mlflow.pyfunc.load_model(model_uri=f"models:/querytime/latest")
#     query_range_response = range_model.predict(pd.DataFrame({"query": [query]}))
#     start, end = extract_timestamps(query_range_response)
#     all_events = retrieve_events(request, start, end)
#     index = faiss(all_events)
#     # TODO: this only works for summarization/retrieval tasks. What about actions?
#     ## If index is dumb
#     relevant_events = index.query(query)
#     return langchain.run(relevant_events, query)


def generate_state_lines(buffer: collections.deque) -> str:
    # TODO: when I get internet see if RAG framework already has a known technique to deal with context chunking
    import tiktoken

    print(f"Buffer has {len(buffer)} events")

    enc = tiktoken.encoding_for_model(
        SUMMARIZER_OAI_MODEL
    )  # TODO: pick it up from the model meta
    MAX_TOKENS = 4000  # TODO: Also pick it up from the model meta and encode the slack into a smarter heuristic
    buffer_lines = []
    prompt_tokens = len(enc.encode(SUMMARIZATION_PROMPT))
    total_tokens = prompt_tokens
    for index, item in enumerate(buffer):
        buffer_line = f"{item}"
        next_tokens = len(enc.encode(buffer_line)) + 1  # \n
        if total_tokens + next_tokens >= MAX_TOKENS:
            _logger.warning(
                f"Only added {index + 1} events to message out of {len(buffer)}"
            )
            break
        buffer_lines.append(buffer_line)
        total_tokens += next_tokens

    state_lines = "\n".join(buffer_lines)
    print(
        f"Generated {total_tokens} token message, {prompt_tokens} from prompt:\n---------\n{state_lines}\n---------"
    )
    return state_lines


def invoke_model(model: mlflow.pyfunc.PyFuncModel, payload: dict, request: Request):
    scenario_name = request.headers.get(SCENARIO_NAME_HEADER)
    return invoke_model_impl(
        model,
        payload,
        scenario_name=scenario_name,
        input_variables={
            "state_lines": generate_state_lines(request.state.state_ring_buffer)
        },
    )


# This logic is obviously wrong, stub impl
@router.get("/")
def list_deployed_models():
    models = mlflow.search_registered_models()
    return {model.name: model.last_updated_timestamp for model in models}


@router.post("/{model_name}/labels/{model_label}/invocations")
def invoke_labeled_model_version(
    model_name: str, model_label: str, payload: dict, request: Request
):
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_label}"
        )
    except mlflow.MlflowException as e:
        _logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading model: {e}") from e
    except ModuleNotFoundError as me:
        import sys

        with open("/tmp/loaded_modules.txt", "w") as f:
            f.write(str(sys.modules))
        _logger.error(
            f"Error loading model: {me}\npath: {sys.path}\nloaded_modules: /tmp/loaded_modules.txt"
        )
        raise HTTPException(
            status_code=400, detail=f"Error loading model: {me}"
        ) from me
    except Exception as e:
        _logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}") from e

    try:
        return invoke_model(model, payload, request)
    except Exception as e:
        _logger.error(f"Error invoking model: {e}")
        raise HTTPException(status_code=500, detail=f"Error invoking model: {e}") from e
