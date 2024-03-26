import collections
import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
)

from .queries import (
    CREATE_SUMMARY_TABLE,
    ENABLE_PGVECTOR,
    CONVERT_TO_HYPERTABLE,
    ENABLE_TIMESCALE,
)

from config import PostgresStoreSettings
from mlmodels import summarize_events


_LOGGER = logging.getLogger(__name__)


def get_connection_string(
    settings: PostgresStoreSettings, include_password: bool = False
) -> str:
    return f"postgresql+asyncpg://{settings.user}:{settings.password.get_secret_value() if include_password else settings.password}@{settings.address}:{settings.port}/{settings.database}"


# Don't need this until we have real models
# async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
# TODO: Go use real models later
# Base = declarative_base()

# def get_session() -> async_sessionmaker[AsyncSession]:
#     return async_session


async def setup_db(settings: PostgresStoreSettings) -> AsyncEngine:
    connection_string = get_connection_string(settings, include_password=True)
    print(f"Using database: {get_connection_string(settings, include_password=False)}")

    engine: AsyncEngine = create_async_engine(connection_string, future=True, echo=True)

    async with engine.begin() as conn:
        await conn.execute(text(ENABLE_TIMESCALE))
        await conn.execute(text(ENABLE_PGVECTOR))
        await conn.execute(text(CREATE_SUMMARY_TABLE))
        await conn.execute(text(CONVERT_TO_HYPERTABLE))
        # Not available for apache licensed version
        # await conn.execute(text(ADD_RETENTION_POLICY))
        await conn.commit()
    return engine


# TODO: move the relevant stuff to rag interface
# TODO: probably rename to mlmodels to reduce confusion with dbmodels


async def insert_summary(
    engine: AsyncEngine,
    include_challenger: bool,
    state_ring_buffer: collections.deque[dict],
):
    print("Inserting summary")
    # TODO: do this better as a batch insert
    # use summarizer model to emit a LIST of summaries, each with the timestamp from relevant event
    events = [json.dumps(event) for event in list(state_ring_buffer)]
    # summarized_events = summarize_events(events)
    champion_summary, challenger_summary = summarize_events(
        ["\n".join(events)], include_challenger
    )
    print(champion_summary)
    print(challenger_summary)
    champion_summary = champion_summary[0]
    challenger_summary = challenger_summary[0]

    # TODO: don't use truncation, use averaging or some other form of dealing with long inputs
    # embedding_vector: list[float] = embed_summary(single_summary)
    async with engine.begin() as conn:
        await conn.execute(
            # text("INSERT INTO summary_data (time, summary, embedding, events) VALUES (NOW(), :summary, :embedding, :events)"),
            # {
            #     "summary": single_summary,
            #     "events": events,
            #     "embedding": str(embedding_vector)
            # }
            text(
                "INSERT INTO summary_data (time, summary, summary_challenger, events) VALUES (NOW(), :summary, :summary_challenger, :events)"
            ),
            {
                "summary": champion_summary,
                "summary_challenger": challenger_summary,
                "events": events,
            },
        )
        await conn.commit()
        _LOGGER.info(f"Inserted summary with {len(state_ring_buffer)} events")
