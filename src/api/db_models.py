# api/db_models.py
# SQLAlchemy models for NeonDB/PostgreSQL
from sqlalchemy import Column, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    page_content = Column(Text)
    metadata = Column(JSON)

# Add more DB models as needed
