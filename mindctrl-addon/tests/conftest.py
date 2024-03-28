from enum import Enum
import logging
import os
from pathlib import Path
import time
import aiomqtt
from pydantic import SecretStr
import pytest
import multiprocessing
from uvicorn import Config, Server
import httpx
import sqlalchemy
import docker
import subprocess

from testcontainers.postgres import PostgresContainer
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.core.container import DockerContainer

from config import AppSettings, MqttEventsSettings, PostgresStoreSettings
from k3d_cluster_manager import LocalRegistryK3dManager
import constants


_logger = logging.getLogger(__name__)


class DeployMode(Enum):
    LOCAL = "local"
    K3D = "k3d"


def pytest_addoption(parser):
    parser.addoption(
        "--deploy-mode",
        action="store",
        default=DeployMode.LOCAL.value,
        help=f"mindctrl deployment mode for integration: {list(DeployMode)}",
    )


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


def build_app(app: str, k3d_registry_url: str, client: docker.DockerClient):
    tag = f"{k3d_registry_url}/{app}:latest"
    app_to_build = Path(__file__).parent.parent / "rootfs/usr/bin" / app

    _logger.info(f"Building {app_to_build} with tag {tag}")
    _, resp = client.images.build(
        path=str(app_to_build),
        tag=tag,
        pull=True,
        rm=True,
        forcerm=True,
    )  # type: ignore

    for line in resp:
        _logger.debug(line)
    return tag


def push_app(tag: str, client: docker.DockerClient):
    _logger.info(f"Pushing {tag}")
    resp = client.images.push(tag, stream=True, decode=True)
    ## WARNING: For some reason, not pulling on the logs will cause the push to fail
    for line in resp:
        _logger.debug(line)


@pytest.fixture(scope="session")
def deploy_mode(request):
    arg = request.config.getoption("--deploy-mode")
    return DeployMode(arg)


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session")
def postgres(deploy_mode: DeployMode):
    if deploy_mode != DeployMode.LOCAL:
        yield None
        return

    _logger.info("Starting postgres fixture")
    postgres = PostgresContainer(
        image="timescale/timescaledb-ha:pg16-all-oss",
        user=constants.POSTGRES_USER,
        password=constants.POSTGRES_PASSWORD,
        dbname=constants.POSTGRES_DB,
    )
    with postgres as p:
        engine = sqlalchemy.create_engine(p.get_connection_url())
        with engine.begin() as connection:
            result = connection.execute(sqlalchemy.text("select version()"))
            (version,) = result.fetchone()  # pyright: ignore
            _logger.info(version)
        yield p


class MosquittoContainer(DockerContainer):
    def __init__(
        self, image="eclipse-mosquitto:latest", port=constants.MQTT_PORT, **kwargs
    ):
        super().__init__(image, **kwargs)
        self.port_to_expose = port
        self.with_exposed_ports(self.port_to_expose)
        self.with_command("mosquitto -c /mosquitto-no-auth.conf")

    def stop(self, force=True, delete_volume=True):
        _logger.info("Stopping mosquitto")
        return super().stop(force, delete_volume)


@pytest.fixture(scope="session")
async def mosquitto(deploy_mode: DeployMode):
    if deploy_mode != DeployMode.LOCAL:
        yield None
        return

    _logger.info("Starting local mosquitto fixture")
    with MosquittoContainer() as mosquitto:
        wait_for_logs(mosquitto, r"mosquitto version [0-9\.]+ running")
        host, port = get_mosquitto_container_host_port(mosquitto)
        async with aiomqtt.Client(hostname=host, port=port) as client:
            assert client._connected.done()
        yield mosquitto


def get_mosquitto_container_host_port(mosquitto: MosquittoContainer) -> tuple[str, int]:
    return (
        mosquitto.get_container_host_ip(),
        int(mosquitto.get_exposed_port(mosquitto.port_to_expose)),
    )


@pytest.fixture(scope="session")
def hosting_settings(mosquitto, postgres, monkeypatch_session, deploy_mode: DeployMode):
    if deploy_mode != DeployMode.LOCAL:
        yield None
        return

    mqtt_host, mqtt_port = get_mosquitto_container_host_port(mosquitto)
    assert isinstance(mqtt_port, int)

    db_url = sqlalchemy.engine.url.make_url(postgres.get_connection_url())

    with monkeypatch_session.context() as m:
        m.setenv("STORE__STORE_TYPE", "psql")
        m.setenv("STORE__USER", db_url.username)
        m.setenv("STORE__PASSWORD", db_url.password)
        m.setenv("STORE__ADDRESS", db_url.host)
        m.setenv(
            "STORE__PORT", str(db_url.port)
        )  # testcontainers spins up on random ports
        m.setenv("STORE__DATABASE", db_url.database)
        m.setenv("EVENTS__EVENTS_TYPE", "mqtt")
        m.setenv("EVENTS__BROKER", mqtt_host)
        m.setenv("EVENTS__PORT", str(mqtt_port))

        # TODO: maybe just take a connection string as a setting instead of exploded
        yield AppSettings(
            store=PostgresStoreSettings(
                user=postgres.POSTGRES_USER,
                password=postgres.POSTGRES_PASSWORD,
                address=constants.POSTGRES_HOST,
                port=constants.POSTGRES_PORT,
                database=postgres.POSTGRES_DB,
                store_type="psql",
            ),
            events=MqttEventsSettings(
                events_type="mqtt", broker=mqtt_host, port=mqtt_port
            ),
            openai_api_key=SecretStr("key"),
        )


def prepare_apps(
    source_dir: Path,
    target_dir: Path,
    registry_url: str,
    docker_client: docker.DockerClient,
):
    _logger.info(
        f"Pulling spec templates from {source_dir}, generating in {target_dir}"
    )

    built_tags = []
    for app in source_dir.glob("*.yaml"):
        target_app = target_dir / app.name

        # Don't push until the registry is created later
        if "postgres" not in app.name and "mosquitto" not in app.name:
            built_tags.append(build_app(app.stem, registry_url, docker_client))

        with open(app, "r") as f:
            content = f.read()
            content = os.path.expandvars(content)
        with open(target_app, "w") as f:
            f.write(content)

    _logger.info(f"Built tags {built_tags}")
    return built_tags


def wait_for_readiness(url: str, attempts: int = 1):
    attempts = 1
    while attempts < constants.MAX_ATTEMPTS:
        try:
            response = httpx.get(url)
            if response.status_code == 200:
                _logger.info(f"Shared multiserver fixture is ready at {url}")
                return
        except httpx.ConnectError as e:
            _logger.debug(f"Waiting for shared multiserver fixture startup..{e}")
        finally:
            attempts += 1
            time.sleep(2)


@pytest.fixture(scope="session")
def k3d_server_url(
    deploy_mode: DeployMode,
    tmp_path_factory: pytest.TempPathFactory,
    docker_client: docker.DockerClient,
    monkeypatch_session,
):
    if deploy_mode != DeployMode.K3D:
        yield None
        return

    _logger.info("Starting k3d cluster")
    cluster = LocalRegistryK3dManager(
        constants.REGISTRY_NAME, constants.REGISTRY_PORT, constants.CLUSTER_NAME
    )
    port_forward_process = None
    try:
        with monkeypatch_session.context() as m:
            m.setenv("K3D_REGISTRY_URL", cluster.k3d_registry_url)

            # this forces my hand: https://github.com/k3d-io/k3d/issues/1418
            cluster.create(
                options=["-p", f'"{constants.K8S_INGRESS_PORT}:80@loadbalancer"']
            )
            cluster.set_kubectl_default()
            cluster.install_dapr()

            m.setenv("STORE__STORE_TYPE", "psql")
            m.setenv("STORE__USER", constants.POSTGRES_USER)
            m.setenv("STORE__PASSWORD", constants.POSTGRES_PASSWORD)
            m.setenv("STORE__ADDRESS", "postgres-service")
            m.setenv("STORE__PORT", str(constants.POSTGRES_PORT))
            m.setenv("STORE__DATABASE", constants.POSTGRES_DB)
            m.setenv("EVENTS__EVENTS_TYPE", "mqtt")
            m.setenv("EVENTS__BROKER", "mosquitto-service")
            m.setenv("EVENTS__PORT", str(constants.MQTT_PORT))
            m.setenv("EVENTS__USERNAME", constants.MQTT_USER)
            m.setenv("EVENTS__PASSWORD", constants.MQTT_PASSWORD)

            target_deploy_folder = tmp_path_factory.mktemp("deploy-resolved")
            deploy_folder = Path(__file__).parent.parent / "rootfs/usr/bin/deploy"
            built_tags = prepare_apps(
                deploy_folder,
                target_deploy_folder,
                cluster.k3d_registry_url,
                docker_client,
            )

            # The registry is created here so now push the images
            for tag in built_tags:
                push_app(tag, docker_client)

            published_images = httpx.get(
                f"http://{cluster.k3d_registry_url}/v2/_catalog"
            ).json()
            _logger.info(f"Published images: {published_images}")

            _logger.info("Creating secrets")
            cluster.create_secret("openai-api-key", "OPENAI_API_KEY")
            cluster.create_secret("store-password", "STORE__PASSWORD")
            cluster.create_secret("events-password", "EVENTS__PASSWORD")

            _logger.info("Applying fixture k8s specs")
            cluster.apply(target_deploy_folder / "mosquitto.yaml")
            cluster.apply(target_deploy_folder / "postgres.yaml")

            cluster.wait_and_get_logs("mosquitto")
            cluster.wait_and_get_logs("postgres")

            _logger.info("Applying app k8s specs")
            cluster.apply(target_deploy_folder / "deployments.yaml")
            cluster.apply(target_deploy_folder / "tracking.yaml")
            cluster.apply(target_deploy_folder / "multiserver.yaml")

            _logger.info("Waiting for deployments to be available")
            cluster.wait_and_get_logs("multiserver", timeout=300)

            port_forward_process = cluster.port_forwarding(
                "service/mosquitto-service", constants.MQTT_PORT, constants.MQTT_PORT
            )
            port_forward_process.start()

            multiserver_url = (
                f"http://{constants.K8S_INGRESS_HOST}:{constants.K8S_INGRESS_PORT}"
            )

            wait_for_readiness(f"{multiserver_url}/version")

            yield multiserver_url

    except subprocess.TimeoutExpired as e:
        _logger.error(f"Timeout waiting for readiness: {e}")
        if not os.environ.get("CI", "false") == "true":
            breakpoint()

    finally:
        _logger.info("Deleting cluster")

        # comment out to keep cluster for debugging, but
        # remember to k3d registry delete AND k3d cluster delete manually
        cluster.delete()
        # Easy to recreate and zombie procs are annoying
        if port_forward_process:
            port_forward_process.stop()


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: Config):
        super().__init__()
        self.server = Server(config=config)
        self.config = config

    def stop(self):
        self.terminate()

    def run(self, *args, **kwargs):
        self.server.run()


@pytest.fixture(scope="session")
def local_server_url(hosting_settings: AppSettings, deploy_mode: DeployMode):
    if deploy_mode != DeployMode.LOCAL:
        yield None
        return

    # For typing
    assert hosting_settings.store.store_type == "psql"
    assert hosting_settings.events.events_type == "mqtt"

    host = constants.LOCAL_MULTISERVER_HOST
    port = constants.LOCAL_MULTISERVER_PORT
    base_url = f"http://{host}:{port}"

    config = Config("main:app", host=host, port=port, log_level="debug")
    server = UvicornServer(config=config)
    _logger.info("Starting shared multiserver fixture")
    server.start()

    wait_for_readiness(base_url)

    yield base_url

    server.stop()


@pytest.fixture
async def server_client(local_server_url, k3d_server_url, deploy_mode: DeployMode):
    match deploy_mode:
        case DeployMode.LOCAL:
            async with httpx.AsyncClient(base_url=local_server_url) as client:
                yield client
        case DeployMode.K3D:
            async with httpx.AsyncClient(base_url=k3d_server_url, timeout=10) as client:
                yield client
        case _:
            raise ValueError(f"Unsupported deploy mode: {deploy_mode}")


@pytest.fixture
async def mqtt_client(hosting_settings, k3d_server_url, deploy_mode: DeployMode):
    match deploy_mode:
        case DeployMode.LOCAL:
            async with aiomqtt.Client(
                hostname=hosting_settings.events.broker,
                port=hosting_settings.events.port,
                username=None,
                password=None,
            ) as client:
                yield client

        case DeployMode.K3D:
            # TODO: Convert fixtures to a base class/settings named tuple return type
            async with aiomqtt.Client(
                hostname=constants.PROXY_MQTT_HOST,
                port=constants.MQTT_PORT,
                username=constants.MQTT_USER,
                password=constants.MQTT_PASSWORD,
            ) as client:
                yield client

        case _:
            raise ValueError(f"Unsupported deploy mode: {deploy_mode}")
