import logging
import time
from pydantic import SecretStr
import pytest
import multiprocessing
from uvicorn import Config, Server
import httpx
import sqlalchemy

from config import AppSettings, MqttEventsSettings, PostgresStoreSettings

_logger = logging.getLogger(__name__)


# @pytest.fixture(scope="session", autouse=True)
# def load_env():
#     load_dotenv()


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session")
def postgres():
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
def hosting_settings(mosquitto, postgres, monkeypatch_session):
    mqtt_host, mqtt_port = mosquitto

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


@pytest.fixture(scope="session")
def shared_server_url(hosting_settings: AppSettings):
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
async def server_client(shared_server_url):
    async with httpx.AsyncClient(base_url=shared_server_url) as client:
        yield client
