TABLE_NAME = "summary_data"
from .models.summary_data import EMBEDDING_DIM

CREATE_TABLE = """CREATE TABLE IF NOT EXISTS {table_name}
(
    "time" timestamp with time zone NOT NULL,
    summary text  NOT NULL,
    embedding vector({embedding_dim}),
    events json[]
)""".format(table_name=TABLE_NAME, embedding_dim=EMBEDDING_DIM)


# https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
ENABLE_PGVECTOR = "CREATE EXTENSION IF NOT EXISTS vector"


# https://docs.timescale.com/self-hosted/latest/install/installation-docker/#setting-up-the-timescaledb-extension
ENABLE_TIMESCALE = "CREATE EXTENSION IF NOT EXISTS timescaledb"


# https://docs.timescale.com/quick-start/latest/python/#create-a-hypertable
CONVERT_TO_HYPERTABLE = """SELECT create_hypertable('{table_name}', by_range('time'), if_not_exists => TRUE)""".format(table_name=TABLE_NAME)


# https://docs.timescale.com/api/latest/data-retention/add_retention_policy/
ADD_RETENTION_POLICY = """SELECT add_retention_policy('{table_name}', drop_after => INTERVAL '2 months')""".format(table_name=TABLE_NAME)
