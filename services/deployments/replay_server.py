# https://github.com/mlflow/mlflow/blob/master/mlflow/deployments/cli.py#L109
# This server helps with AI model testing and debugging by replaying requests to the model
# TODO: mainline this feature into mlflow and drop this file

import functools
from pathlib import Path
import click
from fastapi import Request as fastRequest, HTTPException
import logging
import os
import subprocess
import sys
from typing import Optional, Union
import vcr

## MLflow Patching
from mlflow.deployments.cli import validate_config_path
from mlflow.deployments.server.app import GatewayAPI, create_app_from_path
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_CONFIG
from mlflow.deployments.server.runner import Runner, monitor_config
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat
from mlflow.gateway.utils import make_streaming_response
from mlflow.exceptions import MlflowException
##

## VCR Patching
from yarl import URL
import vcr.stubs.aiohttp_stubs
from vcr.stubs.aiohttp_stubs import (
    play_responses,
    record_responses,
    _build_cookie_header,
    _serialize_headers,
    _build_url_with_params,
)
from vcr.errors import CannotOverwriteExistingCassetteException
from vcr.request import Request
from aiohttp import hdrs

##
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

        # https://slowapi.readthedocs.io/en/latest/#limitations-and-known-issues
        async def _chat(
            request: fastRequest, payload: chat.RequestPayload
        ) -> Union[chat.ResponsePayload, chat.StreamResponsePayload]:
            cassette = Path(capture_directory) / "replay.json"
            _logger.warning(
                f"REPLAY SERVER::Replay mode {replay} for {config.model.name}, patching chat endpoint to cassette {cassette}"
            )

            with vcr.use_cassette(
                cassette,
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
                "replay_server:create_app_from_env()",  # This is the magic to hook into the child processes
            ],
            env={**os.environ, **new_env},
        )


def validate_replay_path(_ctx, _param, cassette_path: Optional[str]):
    # throws, so just invoke it
    if not cassette_path:
        raise click.BadParameter(f"{cassette_path} not provided")

    # Doesn't work for empty file in recording mode
    # try:
    #     FilesystemPersister.load_cassette(cassette_path, jsonserializer)
    #     return cassette_path
    # except Exception as e:
    # if not os.path.exists(cassette_path):
    #     raise click.BadParameter(f"{cassette_path} does not exist")
    return cassette_path


@click.command("serve", help="Start the mindctrl replay server")
@click.option(
    "--config-path",
    envvar=MLFLOW_DEPLOYMENTS_CONFIG.name,
    callback=validate_config_path,
    required=True,
    help="The path to the deployments configuration file.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="The network address to listen on (default: 0.0.0.0).",
)
@click.option(
    "--port",
    default=5001,
    help="The port to listen on (default: 5001).",
)
@click.option(
    "--workers",
    default=1,
    help="The number of workers.",
)
@click.option(
    "--replay-dir",
    envvar="MINDCTRL_REPLAY_DIR",
    required=True,
    help="The path to the vcr casette file.",
)
@click.option(
    "--recording-dir",
    envvar="MINDCTRL_RECORDING_DIR",
    required=True,
    help="The path to the vcr casette file.",
)
@click.option(
    "--replay",
    is_flag=True,
    required=False,
    help="replay or live",
)
def serve(
    config_path: str,
    host: str,
    port: int,
    workers: int,
    replay_dir: str,
    recording_dir: str,
    replay: bool,
):
    click.echo(
        f"Replay server starting on {host}:{port} with {workers} workers with config {config_path} and replay mode {replay} at {replay_dir} and recording to {recording_dir}"
    )
    replay_path = replay_dir if replay else recording_dir
    files = os.listdir(replay_path)
    _logger.warning(f"Replay server starting in replay mode at {', '.join(files)}")

    # https://github.com/mlflow/mlflow/blob/master/mlflow/deployments/server/runner.py#L96
    config_path = os.path.abspath(os.path.normpath(os.path.expanduser(config_path)))
    with ReplayRunner(
        config_path=config_path,
        host=host,
        port=port,
        workers=workers,
        replay_path=replay_path,
        replay=replay,
    ) as runner:
        for _ in monitor_config(config_path):
            _logger.info("Configuration updated, reloading workers")
            runner.reload()

    files = os.listdir(replay_path)
    _logger.warning(f"Replay server finishing in replay mode at {', '.join(files)}")


if __name__ == "__main__":
    serve()
