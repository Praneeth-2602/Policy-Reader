# api/routes.py
# FastAPI endpoints will be moved here.
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status, Depends
from typing import List, Dict, Any, Optional
import os
import logging
from config import settings
from src.loaders.document_loader import DocumentProcessor
from src.utils.text_processing import TextProcessor
from src.utils.vector_store import VectorStoreManager
from src.agents.query_parser import QueryParser
from src.agents.models import ParsedQuery, DecisionResponse, Justification
from src.pipeline.retriever import ClauseRetriever
from src.core.decision_engine import DecisionEngine
from pydantic import BaseModel

# SQLAlchemy imports for DB integration
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import hashlib

# DB setup (reads from config.py or .env)
DATABASE_URL = getattr(settings, 'DATABASE_URL', None)
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from sqlalchemy import ForeignKey
# DB model for processed documents (now also stores extracted text)
class ProcessedDocument(Base):
    __tablename__ = 'processed_documents'
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    file_hash = Column(String, index=True, nullable=False)
    local_path = Column(String, nullable=False)
    extracted_text = Column(String, nullable=True)
    processed_at = Column(DateTime, default=datetime.now)


# Create table if not exists (for demo/hackathon use)
Base.metadata.create_all(bind=engine)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query_parser_instance = QueryParser()
clause_retriever_instance = ClauseRetriever()
decision_engine_instance = DecisionEngine()


class HackRxRunRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]

class ExplainableAnswer(BaseModel):
    answer: str
    clause_snippet: Optional[str] = None
    clause_tag: Optional[str] = None
    justification: Optional[str] = None

class HackRxRunResponse(BaseModel):
    answers: List[ExplainableAnswer]

class RetrievedDocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

@router.get("/")
async def read_root():
    return {"message": "Welcome to the LLM Policy Information System! Use /process-claim to get started."}

@router.post("/parse-query", response_model=ParsedQuery, summary="Parse Natural Language Query")
async def parse_user_query(user_query: UserQuery):
    try:
        parsed_data = await query_parser_instance.parse_query(user_query.query)
        return parsed_data
    except Exception as e:
        logger.error(f"API Error during query parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse query: {e}")

@router.post("/retrieve-clauses", response_model=List[RetrievedDocumentResponse], summary="Retrieve Relevant Clauses")
async def retrieve_relevant_clauses(user_query: UserQuery):
    logger.info(f"Received query for retrieval: '{user_query.query}'")
    try:
        parsed_data = await query_parser_instance.parse_query(user_query.query)
        retrieved_documents = clause_retriever_instance.retrieve_clauses(parsed_data, k=5)
        response_docs = [
            RetrievedDocumentResponse(page_content=doc.page_content, metadata=doc.metadata)
            for doc in retrieved_documents
        ]
        logger.info(f"Returning {len(response_docs)} retrieved documents.")
        return response_docs
    except Exception as e:
        logger.error(f"API Error during retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during clause retrieval: {e}")

@router.post("/upload-document", summary="Upload and ingest a document")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
        file_path = os.path.join(settings.KNOWLEDGE_BASE_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        doc_processor = DocumentProcessor()
        loaded_documents = doc_processor.load_document(file_path)
        if not loaded_documents:
            raise HTTPException(status_code=400, detail="Document could not be loaded or parsed.")
        text_processor = TextProcessor()
        chunks = text_processor.chunk_documents(loaded_documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document could not be chunked.")
        vector_store_manager = VectorStoreManager(text_processor.embeddings_model)
        vector_store_manager.add_documents(chunks)
        logger.info(f"Document '{file.filename}' uploaded, chunked, and stored successfully. Chunks: {len(chunks)}")
        return {"message": f"Document '{file.filename}' uploaded, chunked, and stored successfully.", "chunks_added": len(chunks)}
    except Exception as e:
        logger.error(f"Error during document upload/ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during document upload/ingestion: {e}")


# New endpoint for HackRx batch Q&A
import requests
import tempfile

@router.post("/hackrx/run", response_model=HackRxRunResponse, summary="Batch Q&A on policy document from URL")
async def hackrx_run(request: Request, payload: HackRxRunRequest, db: Session = Depends(get_db)):
    # Optional: Bearer token check (can be made stricter as needed)
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Optionally, check token value here


    # 1. Check if document already processed (by URL)

    doc_url = payload.documents
    processed_doc = db.query(ProcessedDocument).filter_by(url=doc_url).first()
    if processed_doc:
        logger.info(f"Reusing processed document from DB: {processed_doc.local_path}")
        tmp_path = processed_doc.local_path
        extracted_text = processed_doc.extracted_text
        # Check if chunks are already in vector DB (pseudo, replace with actual check if available)
        # If you have a method to check vector DB for doc hash, use it here
        # For now, always re-add chunks (idempotent for most vector DBs)
    else:
        # Download and save
        try:
            response = requests.get(doc_url)
            response.raise_for_status()
            file_hash = hashlib.sha256(response.content).hexdigest()
            os.makedirs(settings.KNOWLEDGE_BASE_DIR, exist_ok=True)
            local_filename = f"policy_{file_hash[:8]}.pdf"
            tmp_path = os.path.join(settings.KNOWLEDGE_BASE_DIR, local_filename)
            with open(tmp_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            logger.error(f"Failed to download document: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
        extracted_text = None

    # 2. Process the document (load, chunk, embed, index)
    try:
        doc_processor = DocumentProcessor()
        loaded_documents = doc_processor.load_document(tmp_path)
        if not loaded_documents:
            raise HTTPException(status_code=400, detail="Document could not be loaded or parsed.")
        # Extract and store text for audit/re-chunking
        extracted_text = "\n".join([doc.page_content for doc in loaded_documents])
        # Save extracted text to DB if not already present
        if not processed_doc:
            new_doc = ProcessedDocument(url=doc_url, file_hash=file_hash, local_path=tmp_path, extracted_text=extracted_text)
            db.add(new_doc)
            db.commit()
            db.refresh(new_doc)
        else:
            processed_doc.extracted_text = extracted_text
            db.commit()
        text_processor = TextProcessor()
        chunks = text_processor.chunk_documents(loaded_documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document could not be chunked.")
        vector_store_manager = VectorStoreManager(text_processor.embeddings_model)
        vector_store_manager.add_documents(chunks)
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during document processing: {e}")

    # 3. For each question, run the pipeline
    answers = []
    for question in payload.questions:
        try:
            parsed_data = await query_parser_instance.parse_query(question)
            retrieved_documents = clause_retriever_instance.retrieve_clauses(parsed_data, k=5)
            final_decision = await decision_engine_instance.evaluate_claim(parsed_data, retrieved_documents)
            # Extract explainable fields
            if hasattr(final_decision, 'answer'):
                answer = final_decision.answer
            elif hasattr(final_decision, 'decision'):
                answer = final_decision.decision
            else:
                answer = str(final_decision)
            # Try to extract clause and justification
            clause_snippet = None
            clause_tag = None
            justification = None
            if hasattr(final_decision, 'justification') and final_decision.justification:
                # If justification is a list of dicts, take the first
                just = final_decision.justification[0] if isinstance(final_decision.justification, list) and final_decision.justification else final_decision.justification
                clause_snippet = just.get('clause_snippet') if isinstance(just, dict) else None
                clause_tag = just.get('clause_tag') if isinstance(just, dict) else None
                justification = just.get('reasoning') if isinstance(just, dict) else str(just)
            tag = getattr(parsed_data, 'tag', None)
            explainable = ExplainableAnswer(
                answer=answer,
                clause_snippet=clause_snippet,
                clause_tag=clause_tag or tag,
                justification=justification
            )
            answers.append(explainable)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            explainable = ExplainableAnswer(answer=f"Error: {e}")
            answers.append(explainable)
    return HackRxRunResponse(answers=answers)
