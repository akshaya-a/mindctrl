from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Tuple
import aiomqtt
from pydantic import SecretStr
import pytest
import httpx
import sqlalchemy
import docker
import subprocess
import shutil

from testcontainers.postgres import PostgresContainer
from testcontainers.core.waiting_utils import wait_for_logs

from mindctrl.config import AppSettings, MqttEventsSettings, PostgresStoreSettings
import constants
from mindctrl.const import REPLAY_SERVER_INPUT_FILE_SUFFIX
from utils.common import (
    build_app,
    dump_container_logs,
    push_app,
    wait_for_readiness,
    get_external_host_port,
)
from utils.local import (
    LocalMultiserver,
    MosquittoContainer,
    DeploymentServerContainer,
)
from utils.cluster import LocalRegistryK3dManager, prepare_apps
from utils.addon import AddonContainer, create_mock_supervisor


# This is gross..
import mlflow.tracking._tracking_service.utils

# TODO: [mlflow] get a real environment variable for this
TEST_ARTIFACT_PATH = "./mlruns"


def _patched_get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    if artifact_uri is None:
        artifact_uri = TEST_ARTIFACT_PATH
    return SqlAlchemyStore(store_uri, artifact_uri)


mlflow.tracking._tracking_service.utils._get_sqlalchemy_store = (
    _patched_get_sqlalchemy_store
)


_logger = logging.getLogger(__name__)


# Test Suite Config
class DeployMode(Enum):
    LOCAL = "local"
    K3D = "k3d"
    ADDON = "addon"


class ReplayMode(Enum):
    LIVE = "live"
    REPLAY = "replay"


def pytest_addoption(parser):
    parser.addoption(
        "--deploy-mode",
        action="store",
        default=DeployMode.LOCAL.value,
        help=f"mindctrl deployment mode for integration: {list(DeployMode)}",
    )
    parser.addoption(
        "--replay-mode",
        action="store",
        default=ReplayMode.REPLAY.value,
        help=f"mindctrl replay mode for integration: {list(ReplayMode)}",
    )


@pytest.fixture(scope="session")
def repo_root_dir():
    root = Path(__file__).parent.parent.resolve()
    _logger.info(f"Executing in {os.getcwd()} with calculated root {root}")
    return root


@pytest.fixture(scope="session")
def test_data_dir(repo_root_dir: Path):
    test_data = repo_root_dir / "tests" / "test_data"
    _logger.info(f"Executing in {os.getcwd()} with calculated test_data {test_data}")
    return test_data


@pytest.fixture(scope="session")
def deploy_mode(request):
    arg = request.config.getoption("--deploy-mode")
    return DeployMode(arg)


@pytest.fixture(scope="session")
def replay_mode(request):
    arg = request.config.getoption("--replay-mode")
    return ReplayMode(arg)


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session")
def postgres(deploy_mode: DeployMode):
    if deploy_mode == DeployMode.K3D:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

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
        dump_container_logs(p, debug=True)


@pytest.fixture(scope="session")
async def mosquitto(deploy_mode: DeployMode):
    if deploy_mode == DeployMode.K3D:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    _logger.info("Starting local mosquitto fixture")
    with MosquittoContainer() as mosquitto:
        wait_for_logs(mosquitto, r"mosquitto version [0-9\.]+ running")
        host, port = get_external_host_port(mosquitto)
        async with aiomqtt.Client(hostname=host, port=port) as client:
            assert client._connected.done(), "Failed to connect to mosquitto"
        yield mosquitto


@dataclass
class ReplayServerExecutionDir:
    replays_dir: Path
    recordings_dir: Path
    config_dir: Path
    server_dir: Path

    def populate_replays(self, source_dir: Path):
        recorded_calls = list(source_dir.glob(f"*{REPLAY_SERVER_INPUT_FILE_SUFFIX}"))
        _logger.info(
            f"Copying recorded calls to replay server: {', '.join(map(str, recorded_calls))}"
        )
        for calls in recorded_calls:
            shutil.copyfile(calls, self.replays_dir / calls.name)

        # pulling on/exhausting the generator in the log statement bit me, hence these aggressive assertions
        assert (
            len(list(self.replays_dir.glob("*"))) > 0
        ), f"No replays found in {self.replays_dir}"

    def save_recordings(self, target_dir: Path):
        files = list(self.recordings_dir.glob("*"))
        if len(files) > 0:
            _logger.info(
                f"Copying recordings to {target_dir}: {', '.join(map(str, files))}"
            )
            shutil.copytree(self.recordings_dir, target_dir, dirs_exist_ok=True)
        else:
            _logger.info(f"No recordings to copy to {target_dir}")


@pytest.fixture(scope="session")
def replay_server_execution_dir(
    tmp_path_factory: pytest.TempPathFactory,
    repo_root_dir: Path,
):
    deployment_server_dir = tmp_path_factory.mktemp("deployments")
    original_deployment_server = repo_root_dir / "services/deployments"
    assert original_deployment_server.exists(), f"Missing {original_deployment_server}"

    shutil.copytree(
        original_deployment_server, deployment_server_dir, dirs_exist_ok=True
    )

    replays = deployment_server_dir / "replays"
    replays.mkdir()

    recordings = deployment_server_dir / "recordings"
    recordings.mkdir()

    config = deployment_server_dir / "config"
    config.mkdir()
    shutil.copyfile(
        deployment_server_dir / "route-config.yaml", config / "route-config.yaml"
    )
    assert (
        config / "route-config.yaml"
    ).exists(), f"Missing {config / 'route-config.yaml'}"

    _logger.info(f"Prepared replay server execution dir at {deployment_server_dir}")

    replay_server_context = ReplayServerExecutionDir(
        replays_dir=replays,
        recordings_dir=recordings,
        config_dir=config,
        server_dir=deployment_server_dir,
    )
    yield replay_server_context


@pytest.fixture(scope="session")
def mlflow_storage(tmp_path_factory: pytest.TempPathFactory):
    if deploy_mode == DeployMode.K3D:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    mlflow_dir = tmp_path_factory.mktemp("mlflow_storage").absolute()
    _logger.info(f"Using MLflow storage at {mlflow_dir}")
    global TEST_ARTIFACT_PATH
    TEST_ARTIFACT_PATH = str(mlflow_dir)
    return mlflow_dir


@pytest.fixture(scope="session")
def mlflow_fluent_session(
    mlflow_storage: Path,
    deployment_server: DeploymentServerContainer,
    replay_mode: ReplayMode,
    deploy_mode: DeployMode,
    monkeypatch_session,
):
    if deploy_mode == DeployMode.K3D:
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    # using file instead of sqlite:///:memory: for post-mortem debugging
    # It's not that slow
    database_path = f"sqlite:///{mlflow_storage}/mlflow.db"
    mlflow.set_tracking_uri(database_path)
    experiment_id = mlflow.create_experiment(
        "test", artifact_location=str(mlflow_storage)
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    with monkeypatch_session.context() as m:
        m.setenv("MLFLOW_DEPLOYMENTS_TARGET", deployment_server.get_base_url())
        m.setenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
        if replay_mode == ReplayMode.REPLAY:
            m.setenv("OPENAI_API_KEY", "DUMMY")
        yield mlflow.get_tracking_uri()


@pytest.fixture(scope="session")
def deployment_server(
    deploy_mode: DeployMode,
    replay_mode: ReplayMode,
    repo_root_dir: Path,
    replay_server_execution_dir: ReplayServerExecutionDir,
    test_data_dir: Path,
):
    if deploy_mode != DeployMode.LOCAL:
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    deployment_server_dir = replay_server_execution_dir.server_dir
    mindctrl_source = repo_root_dir / "python"
    _logger.info(f"Building deployment server with context: {deployment_server_dir}")

    tag = build_app(deployment_server_dir, None, mindctrl_source)

    # Make the mounts
    config = replay_server_execution_dir.config_dir
    recordings = replay_server_execution_dir.recordings_dir
    replays = replay_server_execution_dir.replays_dir
    if replay_mode == ReplayMode.REPLAY:
        replay_server_execution_dir.populate_replays(test_data_dir)

    _logger.info(
        f"Starting deployment server container with tag {tag} and mounts {replays}, {recordings}, {config}"
    )
    with DeploymentServerContainer(
        route_config=config / "route-config.yaml",
        replays_dir=replays,
        recordings_dir=recordings,
        replay_mode=replay_mode == ReplayMode.REPLAY,
        image=tag,
    ) as server:
        external_port = server.get_exposed_port(server.port_to_expose)
        import time

        time.sleep(
            5
        )  # TODO: this should not be needed - why is /health responding and we get disconnects?
        wait_for_readiness(
            f"http://{server.get_container_host_ip()}:{external_port}/health"
        )

        yield server

        if replay_mode == ReplayMode.LIVE:
            replay_server_execution_dir.save_recordings(test_data_dir)


@pytest.fixture(scope="session")
def local_app_settings(
    mosquitto,
    postgres,
    monkeypatch_session,
    deploy_mode: DeployMode,
    replay_mode: ReplayMode,
    deployment_server: DeploymentServerContainer,
):
    if deploy_mode != DeployMode.LOCAL:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    mqtt_host, mqtt_port = get_external_host_port(mosquitto)
    assert isinstance(mqtt_port, int), f"Invalid mqtt port: {mqtt_port}"

    db_url = sqlalchemy.engine.url.make_url(postgres.get_connection_url())

    with monkeypatch_session.context() as m:
        m.setenv(
            "MLFLOW_DEPLOYMENTS_TARGET",
            f"http://{deployment_server.get_container_host_ip()}:{deployment_server.get_exposed_port(deployment_server.port_to_expose)}",
        )
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
        if replay_mode == ReplayMode.REPLAY:
            m.setenv("OPENAI_API_KEY", "DUMMY")

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


@pytest.fixture(scope="session")
def addon_app_settings(
    mosquitto,
    postgres,
    deploy_mode: DeployMode,
    repo_root_dir: Path,
):
    if deploy_mode != DeployMode.ADDON:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    mqtt_host, mqtt_port = get_external_host_port(mosquitto)
    pg_host, pg_port = get_external_host_port(postgres)

    addon_folder = repo_root_dir / "mindctrl-addon"
    tag = build_app(addon_folder, None, None)

    # TODO: maybe just take a connection string as a setting instead of exploded
    yield (
        AppSettings(
            store=PostgresStoreSettings(
                user=postgres.POSTGRES_USER,
                password=postgres.POSTGRES_PASSWORD,
                address=pg_host,
                port=pg_port,
                database=postgres.POSTGRES_DB,
                store_type="psql",
            ),
            events=MqttEventsSettings(
                events_type="mqtt", broker=mqtt_host, port=mqtt_port
            ),
            openai_api_key=SecretStr("DUMMY"),
        ),
        tag,
    )


@pytest.fixture(scope="session")
def k3d_server_url(
    deploy_mode: DeployMode,
    replay_mode: ReplayMode,
    tmp_path_factory: pytest.TempPathFactory,
    docker_client: docker.DockerClient,
    monkeypatch_session,
    repo_root_dir: Path,
    replay_server_execution_dir: ReplayServerExecutionDir,
    test_data_dir: Path,
):
    if deploy_mode != DeployMode.K3D:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    # PV config
    replay_storage = replay_server_execution_dir.replays_dir
    recording_storage = replay_server_execution_dir.recordings_dir
    _logger.info(
        f"Starting k3d cluster with PVs in {replay_storage}, {recording_storage}"
    )
    cluster = LocalRegistryK3dManager(
        constants.REGISTRY_NAME, constants.REGISTRY_PORT, constants.CLUSTER_NAME
    )
    port_forward_process = None
    try:
        with monkeypatch_session.context() as m:
            m.setenv("K3D_REGISTRY_URL", cluster.k3d_registry_url)

            # this forces my hand: https://github.com/k3d-io/k3d/issues/1418
            cluster.create(
                options=[
                    "-p",
                    f'"{constants.K8S_INGRESS_PORT}:80@loadbalancer"',
                    "--volume",
                    f"{replay_storage}:{constants.K3D_REPLAY_PV_LOCATION}@all",
                    "--volume",
                    f"{recording_storage}:{constants.K3D_RECORDING_PV_LOCATION}@all",
                ]
            )
            cluster.set_kubectl_default()

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
            if replay_mode == ReplayMode.REPLAY:
                m.setenv("OPENAI_API_KEY", "DUMMY")

            target_deploy_folder = tmp_path_factory.mktemp("deploy-resolved")
            deploy_folder = repo_root_dir / "services/deploy"
            assert deploy_folder.exists(), (
                f"Missing {deploy_folder}"
            )  # Life easier when I inevitably move stuff around

            mindctrl_source = repo_root_dir / "python"
            services_dir = repo_root_dir / "services"
            built_tags = prepare_apps(
                deploy_folder,
                target_deploy_folder,
                services_dir,
                cluster.k3d_registry_url,
                mindctrl_source,
            )

            # Normally do this earlier but i break the apps all the time, so break sooner
            cluster.install_dapr()

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

            with open(target_deploy_folder / "replay-pv.yaml", "wt") as f:
                f.write(
                    cluster.generate_persistent_volume(
                        # TODO: ARGH - infuriating bug that the *docker* container host path is needed here.
                        # Attach the volume mounts to the cluster object and reuse them to get rid of this
                        # extra constant?
                        "deployments-replay-pv-volume",
                        Path(constants.K3D_REPLAY_PV_LOCATION),
                    )
                )
            with open(target_deploy_folder / "recording-pv.yaml", "wt") as f:
                f.write(
                    cluster.generate_persistent_volume(
                        "deployments-recording-pv-volume",
                        Path(constants.K3D_RECORDING_PV_LOCATION),
                        mode="ReadWriteOnce",
                    )
                )

            _logger.info("Applying PVs")
            cluster.apply(target_deploy_folder / "replay-pv.yaml")
            cluster.apply(target_deploy_folder / "recording-pv.yaml")

            # TODO: Get rid of all this gross path assumptions all across the fixtures

            is_replay_mode = replay_mode == ReplayMode.REPLAY
            if is_replay_mode:
                _logger.info("Copying recorded calls to replay server")
                replay_server_execution_dir.populate_replays(test_data_dir)
                _logger.info(
                    f"Replay mount contents: {', '.join(map(str, replay_storage.iterdir()))}"
                )

            _logger.info("Creating configmap 'mindctrl-config'")
            cluster.create_configmap(
                "mindctrl-config",
                replay_server_execution_dir.config_dir / "route-config.yaml",
                deployment_server_replay=is_replay_mode,
            )
            # _cm = cluster.show_configmap("mindctrl-config")
            # _logger.info(_cm)

            _logger.info("Applying fixture k8s specs")
            cluster.apply(target_deploy_folder / "mosquitto.yaml")
            cluster.apply(target_deploy_folder / "postgres.yaml")
            # Parallelize a couple lower app deployments
            _logger.info("Applying base app k8s specs")
            cluster.apply(target_deploy_folder / "deployments.yaml")
            cluster.apply(target_deploy_folder / "tracking.yaml")

            cluster.wait_and_get_logs("mosquitto")
            cluster.wait_and_get_logs("postgres")
            cluster.wait_and_get_logs("deployments")
            cluster.wait_and_get_logs("tracking")

            _logger.info("Applying multiserver k8s spec")
            cluster.apply(target_deploy_folder / "multiserver.yaml")

            _logger.info("Waiting for deployments to be available")
            cluster.wait_and_get_logs("multiserver", timeout=300)
            # breakpoint()

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
            _logger.error(
                "##################\nBREAKING TO PRESERVE CLUSTER\n####################"
            )
            breakpoint()

    finally:
        _logger.info("Deleting cluster")

        # comment out to keep cluster for debugging, but
        # remember to k3d registry delete AND k3d cluster delete manually
        # breakpoint()
        cluster.delete()
        # Easy to recreate and zombie procs are annoying
        if port_forward_process:
            port_forward_process.stop()


@pytest.fixture(scope="session")
def local_server_url(
    local_app_settings: AppSettings,
    deploy_mode: DeployMode,
):
    if deploy_mode != DeployMode.LOCAL:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    # For typing
    assert local_app_settings.store.store_type == "psql"
    assert local_app_settings.events.events_type == "mqtt"

    with LocalMultiserver as server:
        _logger.info(f"Starting shared multiserver fixture as pid {server.pid}")
        wait_for_readiness(server.url)
        yield server.url


@pytest.fixture(scope="session")
def addon_url(
    addon_app_settings: Tuple[AppSettings, str],
    deploy_mode: DeployMode,
    replay_mode: ReplayMode,
    replay_server_execution_dir: ReplayServerExecutionDir,
    test_data_dir: Path,
):
    if deploy_mode != DeployMode.ADDON:
        raise ValueError(f"Unsupported deploy mode: {deploy_mode}")

    app_settings, tag = addon_app_settings
    # For typing
    assert app_settings.store.store_type == "psql"
    assert app_settings.events.events_type == "mqtt"

    if replay_mode == ReplayMode.REPLAY:
        replay_server_execution_dir.populate_replays(test_data_dir)

    mock_supervisor = create_mock_supervisor(app_settings)
    with mock_supervisor as supervisor:
        _logger.info(f"Starting mock supervisor as pid {supervisor.pid}")
        wait_for_readiness(supervisor.url)

        with AddonContainer(
            supervisor.url,
            replay_server_execution_dir.config_dir / "route-config.yaml",
            replay_server_execution_dir.replays_dir,
            replay_server_execution_dir.recordings_dir,
            replay_mode == ReplayMode.REPLAY,
            image=tag,
        ) as addon:
            wait_for_readiness(addon.get_base_url())
            yield addon.get_base_url()

            if replay_mode == ReplayMode.LIVE:
                replay_server_execution_dir.save_recordings(test_data_dir)


@pytest.fixture(scope="session")
def app_url(
    deploy_mode: DeployMode,
    request: pytest.FixtureRequest,
):
    match deploy_mode:
        case DeployMode.LOCAL:
            return request.getfixturevalue("local_server_url")
        case DeployMode.K3D:
            return request.getfixturevalue("k3d_server_url")
        case DeployMode.ADDON:
            return request.getfixturevalue("addon_url")
        case _:
            raise ValueError(f"Unsupported deploy mode: {deploy_mode}")


# TODO: This might be better done via indirection: https://docs.pytest.org/en/latest/example/parametrize.html#deferring-the-setup-of-parametrized-resources
@pytest.fixture
async def server_client(
    app_url: str,
    request: pytest.FixtureRequest,
):
    headers = {"x-mctrl-scenario-name": request.node.name}
    async with httpx.AsyncClient(
        base_url=app_url, headers=headers, timeout=10
    ) as client:
        yield client


@pytest.fixture
async def mqtt_client(request, deploy_mode: DeployMode):
    match deploy_mode:
        case DeployMode.LOCAL:
            local_app_settings = request.getfixturevalue("local_app_settings")
            async with aiomqtt.Client(
                hostname=local_app_settings.events.broker,
                port=local_app_settings.events.port,
                username=None,
                password=None,
            ) as client:
                yield client

        case DeployMode.ADDON:
            app_settings, _ = request.getfixturevalue("addon_app_settings")
            async with aiomqtt.Client(
                hostname=app_settings.events.broker,
                port=app_settings.events.port,
                username=None,
                password=None,
            ) as client:
                yield client

        case DeployMode.K3D:
            # TODO: Convert fixtures to a base class/settings named tuple return type
            _ = request.getfixturevalue("k3d_server_url")  # Force the deploy
            async with aiomqtt.Client(
                hostname=constants.PROXY_MQTT_HOST,
                port=constants.MQTT_PORT,
                username=constants.MQTT_USER,
                password=constants.MQTT_PASSWORD,
            ) as client:
                yield client

        case _:
            raise ValueError(f"Unsupported deploy mode: {deploy_mode}")
