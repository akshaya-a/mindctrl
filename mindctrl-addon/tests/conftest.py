from enum import Enum
import logging
import os
from pathlib import Path
from subprocess import CalledProcessError
import time
from typing import Optional, Union
from pydantic import SecretStr
import pytest
import multiprocessing
from uvicorn import Config, Server
import httpx
import sqlalchemy
import docker


from testcontainers.postgres import PostgresContainer

from testcontainers.core.container import DockerContainer

from config import AppSettings, MqttEventsSettings, PostgresStoreSettings
from k3d_cluster_manager import LocalRegistryK3dManager


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
    print("Building", app_to_build, "with tag", tag)
    _, resp = client.images.build(
        path=str(app_to_build),
        tag=tag,
        pull=True,
        rm=True,
        forcerm=True,
    )  # type: ignore
    # for line in resp:
    #     print(line)
    return tag


def push_app(tag: str, client: docker.DockerClient):
    print("Pushing", tag)
    _ = client.images.push(tag, stream=True, decode=True)
    # for line in resp:
    #     print(line)


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

    from testcontainers.postgres import PostgresContainer

    _logger.info("Starting postgres fixture")
    postgres = PostgresContainer(
        image="timescale/timescaledb-ha:pg16-all-oss",
        user="test-mindctrl",
        password="test-password",
        dbname="test-mindctrl",
    )
    with postgres as p:
        engine = sqlalchemy.create_engine(p.get_connection_url())
        with engine.begin() as connection:
            result = connection.execute(sqlalchemy.text("select version()"))
            (version,) = result.fetchone()  # pyright: ignore
            print(version)
        yield p


def start_postgres(network_name: Optional[str] = None):
    print("Starting postgres fixture with network", network_name)
    from testcontainers.postgres import PostgresContainer

    _logger.info(f"Starting postgres fixture with network {network_name}")
    postgres = PostgresContainer(
        image="timescale/timescaledb-ha:pg16-all-oss",
        user="test-mindctrl",
        password="test-password",
        dbname="test-mindctrl",
        network=network_name,
    )
    postgres.start()
    engine = sqlalchemy.create_engine(postgres.get_connection_url())
    with engine.begin() as connection:
        result = connection.execute(sqlalchemy.text("select version()"))
        (version,) = result.fetchone()  # pyright: ignore
        print(version)
    yield postgres
    postgres.stop()


class MosquittoContainer(DockerContainer):
    def __init__(self, image="eclipse-mosquitto:latest", port=1883, **kwargs):
        super().__init__(image, **kwargs)
        self.port_to_expose = port
        self.with_exposed_ports(self.port_to_expose)
        self.with_command("mosquitto -c /mosquitto-no-auth.conf")

    def stop(self, force=True, delete_volume=True):
        print("Stopping mosquitto")
        return super().stop(force, delete_volume)


def get_internal_host_port(
    container: Union[PostgresContainer, MosquittoContainer], network_name: str
) -> tuple[str, int]:
    container_obj = container._container
    assert container_obj is not None
    container_id = container_obj.id
    container_inspect = container.get_docker_client().get_container(container_id)
    print(container_inspect)
    return container_inspect["NetworkSettings"]["Networks"][network_name][
        "IPAddress"
    ], container.port_to_expose


def start_mosquitto(network_name: Optional[str] = None):
    print("Starting mosquitto fixture with network", network_name)
    # # TODO rip all this out for testcontainers if this works
    # from pytest_mqtt.mosquitto import is_mosquitto_running, Mosquitto

    # _logger.info(f"Starting mosquitto fixture with network {network_name}")

    # # Gracefully skip spinning up the Docker container if Mosquitto is already running.
    # if is_mosquitto_running():
    #     yield "localhost", 1883
    #     return

    # mosquitto_image = Mosquitto()
    # if network_name is not None:
    #     mosquitto_image.base_image_options["network"] = network_name

    # # Spin up Mosquitto container.
    # if os.environ.get("MOSQUITTO"):
    #     yield os.environ["MOSQUITTO"].split(":")
    # else:
    #     yield mosquitto_image.run()
    #     mosquitto_image.stop()

    _logger.info(f"Starting mosquitto fixture with network {network_name}")
    mosquitto = MosquittoContainer(network=network_name)
    mosquitto.start()
    # yield (
    #     mosquitto.get_container_host_ip(),
    #     mosquitto.get_exposed_port(mosquitto.port_to_expose),
    # )
    yield mosquitto
    mosquitto.stop()


def get_mosquitto_host_port(mosquitto: MosquittoContainer) -> tuple[str, int]:
    return (
        mosquitto.get_container_host_ip(),
        int(mosquitto.get_exposed_port(mosquitto.port_to_expose)),
    )


@pytest.fixture(scope="session")
def hosting_settings(postgres, monkeypatch_session, deploy_mode: DeployMode):
    if deploy_mode != DeployMode.LOCAL:
        yield None
        return

    mosquitto = next(start_mosquitto())
    mqtt_host, mqtt_port = get_mosquitto_host_port(mosquitto)
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
        # m.setenv("OPENAI_API_KEY", "key")

        # TODO: maybe just take a connection string as a setting instead of exploded
        yield AppSettings(
            store=PostgresStoreSettings(
                user=postgres.POSTGRES_USER,
                password=postgres.POSTGRES_PASSWORD,
                address="localhost",
                port=5432,
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
    print("Preparing apps from", os.getcwd())
    print("Pulling spec templates from", source_dir)
    print("Generating specs in", target_dir)

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

    return built_tags


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

    print("Starting k3d cluster")
    cluster = LocalRegistryK3dManager("ptmctrlreg.localhost", 12345, "mindctrl")
    try:
        with monkeypatch_session.context() as m:
            m.setenv("K3D_REGISTRY_URL", cluster.k3d_registry_url)

            # Host networking to try to talk to testcontainers
            # why not deploy them into the cluster? Good question.
            # Maybe this forces my hand: https://github.com/k3d-io/k3d/issues/1418
            # cluster.create(
            #     options=["--network", "host"],
            #     registry_options=["--default-network", "host"],
            # )
            cluster.create()

            # breakpoint()

            # docker_network_name = f"k3d-{cluster.cluster_name}"
            # postgres = next(start_postgres(docker_network_name))
            # internal_address, internal_port = get_internal_host_port(
            #     postgres, docker_network_name
            # )
            # db_url = sqlalchemy.engine.url.make_url(postgres.get_connection_url())

            # # breakpoint()

            # mosquitto = next(start_mosquitto(docker_network_name))
            # mqtt_host, mqtt_port = get_internal_host_port(
            #     mosquitto, docker_network_name
            # )

            # breakpoint()

            m.setenv("STORE__STORE_TYPE", "psql")
            m.setenv("STORE__USER", "test-user")
            # m.setenv("STORE__USER", db_url.username)
            m.setenv("STORE__PASSWORD", "test-password")
            # m.setenv("STORE__PASSWORD", db_url.password)
            # m.setenv("STORE__ADDRESS", db_url.host)
            # m.setenv("STORE__ADDRESS", internal_address)
            m.setenv("STORE__ADDRESS", "postgres")
            # m.setenv(
            #     "STORE__PORT", str(db_url.port)
            # )  # testcontainers spins up on random ports
            m.setenv(
                "STORE__PORT", str(5432)
            )  # testcontainers spins up on random ports
            m.setenv("STORE__DATABASE", "test-mindctrl")
            # m.setenv("STORE__DATABASE", db_url.database)
            m.setenv("EVENTS__EVENTS_TYPE", "mqtt")
            m.setenv("EVENTS__BROKER", "mosquitto")
            m.setenv("EVENTS__PORT", str(1883))

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

            print("Creating secrets")
            cluster.create_secret("openai-api-key", "OPENAI_API_KEY")
            cluster.create_secret("store-password", "STORE__PASSWORD")
            # cluster.create_secret("events-password", "EVENTS__PASSWORD")

            print("Applying k8s specs")
            cluster.apply(target_deploy_folder / "mosquitto.yaml")
            cluster.apply(target_deploy_folder / "postgres.yaml")
            cluster.apply(target_deploy_folder / "deployments.yaml")
            cluster.apply(target_deploy_folder / "tracking.yaml")
            cluster.apply(target_deploy_folder / "multiserver.yaml")

            # breakpoint()
            print("Waiting for deployments to be created + logs populated")
            import time

            time.sleep(60)
            print(
                cluster.kubectl(
                    ["describe", "pod", "-l", "app=deployments"], as_dict=False
                )
            )
            print(
                cluster.kubectl(
                    ["describe", "pod", "-l", "app=tracking"], as_dict=False
                )
            )
            print(
                cluster.kubectl(
                    ["describe", "pod", "-l", "app=multiserver"], as_dict=False
                )
            )

            print(cluster.kubectl(["logs", "-l", "app=deployments"], as_dict=False))
            print(cluster.kubectl(["logs", "-l", "app=tracking"], as_dict=False))
            print(cluster.kubectl(["logs", "-l", "app=multiserver"], as_dict=False))

            print("Waiting for deployments to be available")
            cluster.wait("deployments/multiserver", "condition=Available=True")
            # this is wrong, get multiserver ingress instead
            yield cluster.k3d_registry_url
    except CalledProcessError as e:
        print(e)
        print(e.cmd)
        print(e.stdout)
        print(e.stderr)
        print(e.output)
        # breakpoint()
    finally:
        print("Deleting cluster")
        # breakpoint()
        # comment out to keep cluster for debugging, but
        # remember to k3d registry delete AND k3d cluster delete

        cluster.delete()

        # comment in for easier debugging
        # cluster._exec(
        #     [
        #         "kubeconfig",
        #         "merge",
        #         cluster.cluster_name,
        #         "--kubeconfig-merge-default",
        #         "--kubeconfig-switch-context",
        #     ]
        # )
        # breakpoint()


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

    host = "127.0.0.1"
    port = 5002
    base_url = f"http://{host}:{port}"

    config = Config("main:app", host=host, port=port, log_level="debug")
    server = UvicornServer(config=config)
    _logger.info("Starting shared multiserver fixture")
    server.start()
    max_attempts = 20
    attempts = 1
    while attempts < max_attempts:
        try:
            httpx.get(base_url)
            break
        except httpx.ConnectError:
            print("Waiting for shared multiserver fixture startup..")
            attempts += 1
            time.sleep(2)

    yield base_url

    server.stop()


@pytest.fixture
async def server_client(local_server_url, k3d_server_url, deploy_mode: DeployMode):
    match deploy_mode:
        case DeployMode.LOCAL:
            async with httpx.AsyncClient(base_url=local_server_url) as client:
                yield client
        case DeployMode.K3D:
            async with httpx.AsyncClient(base_url=k3d_server_url) as client:
                yield client
        case _:
            raise ValueError(f"Unsupported deploy mode: {deploy_mode}")
