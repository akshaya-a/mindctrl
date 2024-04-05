import logging
import multiprocessing
from pathlib import Path
import time
from typing import Iterator, Optional, Union
import httpx
from python_on_whales import docker as docker_cli
from docker import DockerClient
from uvicorn import Config, Server
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer

import constants


_logger = logging.getLogger(__name__)


def build_app(
    app_to_build: Path,
    k3d_registry_url: Optional[str],
    source_path: Optional[Path],
):
    assert app_to_build.exists(), f"Missing {app_to_build}"
    app = app_to_build.stem
    tag = f"{k3d_registry_url}/{app}:latest" if k3d_registry_url else f"{app}:latest"

    _logger.info(f"Building {app_to_build} with tag {tag}")
    contexts = {}
    if source_path:
        assert source_path.exists(), f"Missing {source_path}"
        contexts = {"mindctrl_source": source_path}
    # https://www.docker.com/blog/dockerfiles-now-support-multiple-build-contexts/
    logs: Iterator[str] = docker_cli.build(
        app_to_build,
        build_contexts=contexts,  # type: ignore
        tags=tag,
        pull=True,
        stream_logs=True,
    )  # type: ignore
    for log in logs:
        _logger.debug(log)

    return tag


# TODO: switch this over to python_on_whales too
def push_app(tag: str, client: DockerClient):
    _logger.info(f"Pushing {tag}")
    resp = client.images.push(tag, stream=True, decode=True)
    ## WARNING: For some reason, not pulling on the logs will cause the push to fail
    for line in resp:
        _logger.debug(line)


def wait_for_readiness(url: str, max_attempts=constants.MAX_ATTEMPTS):
    _logger.info(f"Waiting for fixture startup at {url}...........")
    attempts = 1
    while attempts <= max_attempts:
        try:
            response = httpx.get(url)
            if response.status_code == 200:
                _logger.info(f"fixture is ready at {url}")
                return
        except httpx.ConnectError as e:
            _logger.debug(f"Waiting for fixture startup at {url}...{e}")
        except httpx.ReadError as e:
            _logger.debug(f"Waiting for fixture startup at {url}...{e}")
        finally:
            attempts += 1
            time.sleep(2)

    if attempts > max_attempts:
        raise RuntimeError(f"Failed to reach {url} after {max_attempts} attempts")


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: Config):
        super().__init__()
        self.server = Server(config=config)
        self.config = config

    @property
    def url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def stop(self):
        self.terminate()

    def run(self, *args, **kwargs):
        self.server.run()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def dump_container_logs(container: DockerContainer):
    out, err = container.get_logs()
    out_lines = out.decode("utf-8").split("\n")
    err_lines = err.decode("utf-8").split("\n")
    container_logger = logging.getLogger(
        str(container.__class__.__name__).removesuffix("Container")
    )
    container_logger.info(
        f"Logs for {container.__class__.__name__}\n*******************\n"
    )
    for line in out_lines:
        container_logger.info(line)
    for line in err_lines:
        container_logger.info(line)
    container_logger.info("*******************\n")


class ServiceContainer(DockerContainer):
    def __init__(self, image, port, **kwargs):
        super().__init__(image, **kwargs)
        self.port_to_expose = port
        self.host_network_mode = kwargs.get("network_mode") == "host"
        if not self.host_network_mode:
            self.with_exposed_ports(self.port_to_expose)

    def get_base_url(self):
        if self.host_network_mode:
            return f"http://localhost:{self.port_to_expose}"
        return f"http://{self.get_container_host_ip()}:{self.get_exposed_port(self.port_to_expose)}"

    def stop(self, force=True, delete_volume=True):
        _logger.info(f"Stopping {self.__class__.__name__}")
        dump_container_logs(self)
        return super().stop(force, delete_volume)


def get_external_host_port(
    container: Union[ServiceContainer, PostgresContainer],
) -> tuple[str, int]:
    return (
        container.get_container_host_ip(),
        int(container.get_exposed_port(container.port_to_expose)),
    )