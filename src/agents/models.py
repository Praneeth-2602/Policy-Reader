# src/agents/models.py
from pydantic import BaseModel, Field, ConfigDict # Added ConfigDict for future (optional) use
from typing import Optional, List, Dict, Any # Ensure Dict and Any are imported

class ParsedQuery(BaseModel):
    """
    Represents the structured data extracted from a natural language insurance query.
    """
    age: Optional[int] = Field(
        None, description="Age of the individual in years."
    )
    gender: Optional[str] = Field(
        None, description="Gender of the individual (e.g., 'male', 'female', 'other')."
    )
    procedure: Optional[str] = Field(
        None, description="Type of medical procedure or surgery (e.g., 'knee surgery', 'heart bypass')."
    )
    location: Optional[str] = Field(
        None, description="Geographic location where the procedure is to be performed (e.g., 'Pune', 'Mumbai')."
    )
    policy_duration_months: Optional[int] = Field(
        None, description="Duration of the insurance policy in months since its inception."
    )
    keywords: Optional[List[str]] = Field(
        None, description="Additional relevant keywords or phrases from the query not captured by other fields."
    )
    # Using model_config for Pydantic V2 compatibility, replacing class Config:
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 46,
            "gender": "male",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration_months": 3,
            "keywords": ["new policy"]
        }
    })

# --- RetrievedDocumentResponse MUST be defined here ---
class RetrievedDocumentResponse(BaseModel):
    """
    Represents a retrieved document snippet with its content and metadata.
    This model is used for the output of the retrieval tool and as input to the decision tool.
    """
    page_content: str
    metadata: Dict[str, Any] # Use Dict and Any from typing

    # Optional: Example for Swagger UI, using Pydantic V2 style
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "page_content": "Section 2: Medical Procedures. Knee surgery is covered after a 3-month waiting period.",
            "metadata": {"source": "policy_example.txt", "page_number": 1}
        }
    })


class Justification(BaseModel):
    """
    Details explaining the decision, linking to specific clauses.
    """
    clause_snippet: str = Field(
        ..., description="A snippet of the specific clause(s) from the document that informed the decision."
    )
    clause_reference: str = Field(
        ..., description="Reference to the exact location of the clause (e.g., 'Section 4.2, Policy Doc', 'Email dated 2023-01-15')."
    )
    reasoning: str = Field(
        ..., description="Step-by-step reasoning explaining how the decision was derived from the query and the clause(s)."
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "clause_snippet": "Knee surgeries are covered under policy conditions after 2 months of initiation",
            "clause_reference": "Section 4.2, Policy Doc",
            "reasoning": "Query conditions satisfy all requirements (duration > 2 months, procedure included)"
        }
    })


class DecisionResponse(BaseModel):
    """
    The final structured response containing the decision, amount, and justification.
    """
    decision: str = Field(
        ..., description="The final decision (e.g., 'approved', 'rejected', 'requires more information')."
    )
    amount: Optional[float] = Field(
        None, description="The payout amount, if applicable and approved."
    )
    justification: List[Justification] = Field(
        ..., description="A list of justifications, each mapping to a specific clause used for the decision."
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "decision": "approved",
            "amount": 50000.0,
            "justification": [
                {
                    "clause_snippet": "Knee surgeries are covered under policy conditions after 2 months of initiation",
                    "clause_reference": "Section 4.2, Policy Doc",
                    "reasoning": "Query conditions satisfy all requirements (duration > 2 months, procedure included)"
                }
            ]
        }
    })