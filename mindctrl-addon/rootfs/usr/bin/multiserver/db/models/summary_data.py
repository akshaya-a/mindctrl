import datetime

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.postgresql.types import TIMESTAMP
from pgvector.sqlalchemy import Vector


EMBEDDING_DIM = 384

# from ..config import Base

# class SummaryData(Base):
#     __tablename__ = 'summary_data'

#     time = Column(TIMESTAMP, default=datetime.datetime.now, primary_key=True)
#     summary = Column(String, nullable=False)
#     embedding = Column(Vector(EMBEDDING_DIM))
