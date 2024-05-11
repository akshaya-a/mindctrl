import logging
import multiprocessing
from pathlib import Path
import time
from typing import Iterator, Optional, Union
import httpx
from python_on_whales import docker as docker_cli
from docker import DockerClient
import socket
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
            if response.status_code == 200 or response.status_code == 302:
                _logger.info(f"fixture is ready at {url}")
                return
            elif response.status_code >= 400 and response.status_code < 500:
                raise ValueError(f"Failed to reach {url}:\n{response}\n{response.text}")
        except (
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.ReadTimeout,
        ) as e:
            _logger.debug(f"Waiting for fixture startup at {url}...{e}")
        finally:
            attempts += 1
            time.sleep(2)

    if attempts > max_attempts:
        raise RuntimeError(f"Failed to reach {url} after {max_attempts} attempts")


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: Config, wait_suffix: Optional[str] = None):
        super().__init__()
        self.server = Server(config=config)
        self.config = config
        self.wait_suffix = wait_suffix

    @property
    def url(self):
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def healthcheck_url(self):
        suffix = self.wait_suffix or ""
        suffix = suffix.lstrip("/")
        return f"{self.url}/{suffix}"

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


def dump_container_logs(container: DockerContainer, debug=False):
    out, err = container.get_logs()
    out_lines = out.decode("utf-8").split("\n")
    err_lines = err.decode("utf-8").split("\n")
    container_logger = logging.getLogger(
        str(container.__class__.__name__).removesuffix("Container")
    )
    log = container_logger.debug if debug else container_logger.info
    log(f"stdout for {container.__class__.__name__}\n*******************\n")
    for line in out_lines:
        log(line)
    log(f"stderr for {container.__class__.__name__}\n*******************\n")
    for line in err_lines:
        log(line)
    log("*******************\n")


class ServiceContainer(DockerContainer):
    def __init__(self, image, port, log_debug=False, **kwargs):
        super().__init__(image, **kwargs)
        self.port_to_expose = port
        self.host_network_mode = kwargs.get("network_mode") == "host"
        if not self.host_network_mode:
            self.with_exposed_ports(self.port_to_expose)
        self.log_debug = log_debug

    def get_base_url(self):
        if self.host_network_mode:
            return f"http://localhost:{self.port_to_expose}"
        return f"http://{self.get_container_host_ip()}:{self.get_exposed_port(self.port_to_expose)}"

    def stop(self, force=True, delete_volume=True):
        _logger.info(f"Stopping {self.__class__.__name__}")
        dump_container_logs(self, self.log_debug)
        return super().stop(force, delete_volume)


class HAContainer(ServiceContainer):
    def __init__(self, config_dir: Path, **kwargs):
        super().__init__(
            "ghcr.io/home-assistant/home-assistant:stable", port=8123, **kwargs
        )
        self.with_env("TZ", "America/Los_Angeles")
        self.with_kwargs(privileged=True)
        self.with_volume_mapping(str(config_dir), "/config", "rw")


def get_external_host_port(
    container: Union[ServiceContainer, PostgresContainer],
) -> tuple[str, int]:
    return (
        container.get_container_host_ip(),
        int(container.get_exposed_port(container.port_to_expose)),
    )


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(0)
        try:
            # doesn't even have to be reachable
            s.connect(("10.254.254.254", 1))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
