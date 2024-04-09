import logging
import os
from typing import Optional
import click

from mlflow.deployments.cli import validate_config_path
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_CONFIG
from mlflow.deployments.server.runner import monitor_config

from .replay_server import ReplayRunner


_logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


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


# TODO: Tackle the cli second level refactor later
# mindctrl proxy serve
# mindctrl {} create
# ...


@cli.command("serve", help="Start the mindctrl replay server")
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
