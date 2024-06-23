# https://github.com/mlflow/mlflow/blob/master/mlflow/deployments/cli.py#L109
# This server helps with AI model testing and debugging by replaying requests to the model
# TODO: mainline this feature into mlflow and drop this file

import functools
import logging
import os
import subprocess
import sys
from typing import List, Literal, Optional, Union

from pydantic import Field
import vcr
import vcr.stubs.aiohttp_stubs
from aiohttp import hdrs
from fastapi import HTTPException
from fastapi import Request as fastRequest

## MLflow Patching
from mlflow.deployments.server.app import GatewayAPI, create_app_from_path
from mlflow.deployments.server.runner import Runner
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_CONFIG
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat
from mlflow.gateway.utils import make_streaming_response
from vcr.errors import CannotOverwriteExistingCassetteException
from vcr.request import Request
from vcr.stubs.aiohttp_stubs import (
    _build_cookie_header,
    _build_url_with_params,
    _serialize_headers,
    play_responses,
    record_responses,
)

##
## VCR Patching
from yarl import URL

##
from .const import (
    REPLAY_SERVER_INPUT_FILE_SUFFIX,
    REPLAY_SERVER_OUTPUT_FILE_SUFFIX,
    SCENARIO_NAME_HEADER,
)

_logger = logging.getLogger(__name__)
log = _logger  # vcr patch


# Need this PR to be merged to make this work natively
# https://github.com/kevin1024/vcrpy/pull/768/files#diff-de735fe76bf327f1c3b46e9d5a72a406ee3e2b4dd6fb53345b4177d9b6315e5b
def patched_vcr_request(cassette, real_request):
    @functools.wraps(real_request)
    async def new_request(self, method, url, **kwargs):
        headers = kwargs.get("headers")
        auth = kwargs.get("auth")
        headers = self._prepare_headers(headers)
        data = kwargs.get(
            "json", kwargs.get("data")
        )  # <- this line is the only change in the PR that matters
        params = kwargs.get("params")
        cookies = kwargs.get("cookies")

        if auth is not None:
            headers["AUTHORIZATION"] = auth.encode()
        request_url = URL(url) if not params else _build_url_with_params(url, params)
        c_header = headers.pop(hdrs.COOKIE, None)
        cookie_header = _build_cookie_header(self, cookies, c_header, request_url)
        if cookie_header:
            headers[hdrs.COOKIE] = cookie_header
        vcr_request = Request(
            method, str(request_url), data, _serialize_headers(headers)
        )
        if cassette.can_play_response_for(vcr_request):
            log.info(f"Playing response for {vcr_request} from cassette")
            response = play_responses(cassette, vcr_request, kwargs)
            for redirect in response.history:
                self._cookie_jar.update_cookies(redirect.cookies, redirect.url)
            self._cookie_jar.update_cookies(response.cookies, response.url)
            return response
        if cassette.write_protected and cassette.filter_request(vcr_request):
            raise CannotOverwriteExistingCassetteException(
                cassette=cassette, failed_request=vcr_request
            )
        log.info("%s not in cassette, sending to real server", vcr_request)
        response = await real_request(self, method, url, **kwargs)
        await record_responses(cassette, vcr_request, response)
        return response

    return new_request


vcr.stubs.aiohttp_stubs.vcr_request = patched_vcr_request

## VCR Patch


def scrub_oai_response_headers(response):
    response["headers"]["openai-organization"] = "FAKE_OAI_ORG"
    response["headers"]["Set-Cookie"] = "FAKE_OAI_COOKIE"
    return response


def create_app_from_env() -> GatewayAPI:
    capture_directory = os.environ["MINDCTRL_REPLAY_PATH"]
    replay = os.environ.get("MINDCTRL_REPLAY", "false").lower() == "true"
    _logger.warning(
        f"Replay Server::create_app_from_env is hooking the webserver with replay mode {replay} and replay path {capture_directory}"
    )

    recording_mode = "none" if replay else "all"

    ### START THE MONKEY BUSINESS - patches following
    # https://github.com/mlflow/mlflow/blob/master/mlflow/deployments/server/app.py#L80
    def _create_replay_chat_endpoint(config: RouteConfig):
        _logger.warning(
            f"REPLAY SERVER::Replay mode {replay} for {config.model.name}, patching chat endpoint"
        )
        prov = get_provider(config.model.provider)(config)  # type: ignore

        # import mindctrl
        # TODO: define __version__
        # mctrl_header = {"User-Agent": f"mindctrl/{mindctrl.__version__}"}
        mctrl_header = {"User-Agent": "mindctrl/0.1.0"}

        from mlflow.gateway.base_models import ResponseModel, RequestModel
        from mlflow.gateway.schemas.chat import (
            BaseRequestPayload,
            _REQUEST_PAYLOAD_EXTRA_SCHEMA,
        )
        from mlflow.gateway.providers.utils import send_request

        class RequestMessage(RequestModel):
            role: str
            content: Optional[str] = None
            tool_call_id: Optional[str] = None
            name: Optional[str] = None

        class RequestPayload(BaseRequestPayload):
            messages: List[RequestMessage] = Field(..., min_length=1)

            class Config:
                json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA

        class Function(ResponseModel):
            name: str
            arguments: str

        class ToolCall(ResponseModel):
            type: str
            function: Function
            id: str

        class ResponseMessage(ResponseModel):
            tool_calls: Optional[list[ToolCall]]
            role: str
            content: Optional[str] = None

        class Choice(ResponseModel):
            index: int
            message: ResponseMessage
            finish_reason: Optional[str] = None

        class ChatUsage(ResponseModel):
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            total_tokens: Optional[int] = None

        class ResponsePayload(ResponseModel):
            id: Optional[str] = None
            object: Literal["chat.completion"] = "chat.completion"
            created: int
            model: str
            choices: List[Choice]
            usage: ChatUsage

        async def chat_with_tools(self, payload):
            from fastapi.encoders import jsonable_encoder

            print("AI REQUEST", payload)
            payload = jsonable_encoder(payload, exclude_none=True)
            self.check_for_model_field(payload)
            all_headers = {**self._request_headers, **mctrl_header}
            resp = await send_request(
                headers=all_headers,
                base_url=self._request_base_url,
                path="chat/completions",
                payload=self._add_model_to_payload_if_necessary(payload),
            )
            print("AI RESPONSE", resp)

            return ResponsePayload(
                id=resp["id"],
                object=resp["object"],
                created=resp["created"],
                model=resp["model"],
                choices=[
                    Choice(
                        index=idx,
                        message=ResponseMessage(
                            role=c["message"]["role"],
                            content=c["message"].get("content", ""),
                            tool_calls=c["message"].get("tool_calls"),  # type: ignore
                        ),
                        finish_reason=c["finish_reason"],
                    )
                    for idx, c in enumerate(resp["choices"])
                ],
                usage=ChatUsage(
                    prompt_tokens=resp["usage"]["prompt_tokens"],
                    completion_tokens=resp["usage"]["completion_tokens"],
                    total_tokens=resp["usage"]["total_tokens"],
                ),
            )

        import mlflow.gateway.schemas.chat

        mlflow.gateway.schemas.chat.RequestMessage = RequestMessage
        mlflow.gateway.schemas.chat.RequestPayload = RequestPayload
        mlflow.gateway.schemas.chat.ResponseMessage = ResponseMessage
        mlflow.gateway.schemas.chat.ResponsePayload = ResponsePayload
        mlflow.gateway.schemas.chat.Choice = Choice
        mlflow.gateway.schemas.chat.ChatUsage = ChatUsage

        import mlflow.gateway.providers.openai

        mlflow.gateway.providers.openai.OpenAIProvider.chat = chat_with_tools

        # https://slowapi.readthedocs.io/en/latest/#limitations-and-known-issues
        async def _chat(
            request: fastRequest, payload: chat.RequestPayload
        ) -> Union[chat.ResponsePayload, chat.StreamResponsePayload]:
            _logger.info(f"REPLAY SERVER::Chat endpoint with headers {request.headers}")
            filename_suffix = (
                REPLAY_SERVER_INPUT_FILE_SUFFIX
                if replay
                else REPLAY_SERVER_OUTPUT_FILE_SUFFIX
            )
            filename_prefix = (
                f"{request.headers[SCENARIO_NAME_HEADER]}-"
                if SCENARIO_NAME_HEADER in request.headers
                else ""
            )
            filename = f"{filename_prefix}{filename_suffix}"
            _logger.warning(
                f"REPLAY SERVER::Replay mode {replay} for {config.model.name}, "
                f"patching chat endpoint to cassette dir {capture_directory} and filename {filename}"
            )

            with vcr.use_cassette(
                path=filename,
                cassette_library_dir=capture_directory,
                serializer="json",
                record_mode=recording_mode,
                # Warning, this is only for requests
                filter_headers=[("Authorization", "FAKE_BEARER")],
                before_record_response=scrub_oai_response_headers,
                # Default behavior doesn't match on body
                match_on=["method", "scheme", "host", "port", "path", "query", "body"],
            ):
                try:
                    if payload.stream:
                        return await make_streaming_response(prov.chat_stream(payload))  # type: ignore
                    else:
                        return await prov.chat(payload)
                except CannotOverwriteExistingCassetteException as e:
                    _logger.error(f"Replay server couldn't send request:\n{e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Replay server is in replay mode but a new request was found or not properly matched:\n{payload}\n{e}",
                    )

        return _chat

    import mlflow.deployments.server.app

    mlflow.deployments.server.app._create_chat_endpoint = _create_replay_chat_endpoint

    ### END THE MONKEY BUSINESS

    if config_path := MLFLOW_DEPLOYMENTS_CONFIG.get():
        return create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_DEPLOYMENTS_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )


class ReplayRunner(Runner):
    def __init__(
        self,
        config_path: str,
        host: str,
        port: int,
        workers: int,
        replay_path: str,
        replay: bool,
    ):
        super().__init__(config_path, host, port, workers)
        self.process = None
        self.replay_path = replay_path
        self.replay = replay

    def start(self) -> None:
        _logger.warning(
            f"ReplayRunner::start with replay mode {self.replay} at {self.replay_path}"
        )
        new_env = {
            MLFLOW_DEPLOYMENTS_CONFIG.name: self.config_path,
            "MINDCTRL_REPLAY_PATH": self.replay_path,
        }
        if self.replay:
            new_env["MINDCTRL_REPLAY"] = str(self.replay)

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--bind",
                f"{self.host}:{self.port}",
                "--workers",
                str(self.workers),
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "mindctrl.replay_server:create_app_from_env()",  # This is the magic to hook into the child processes
            ],
            env={**os.environ, **new_env},
        )
