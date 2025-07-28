# api/models.py
# Pydantic models for request/response schemas
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class UserQuery(BaseModel):
    query: str

class RetrievedDocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

# Add more models as needed
