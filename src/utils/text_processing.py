# src/utils/text_processing.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tagging additions ---
FIXED_TAGS = [
    "coverage", "exclusion", "waiting_period", "room_rent", "definition", "network_hospital", "maternity", "ncd", "ayush", "sub_limit", "grace_period", "general_info"
]

class TextProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""] # Prioritize splitting by paragraphs, then lines, then words
        )
        # Initialize Google Gemini Embeddings
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY
        )
        logger.info(f"Initialized embedding model: {settings.EMBEDDING_MODEL_NAME}")
        # Import your LLM wrapper here (assume GeminiLLM)
        try:
            from src.agents.models import GeminiLLM
            self.llm = GeminiLLM()
        except Exception:
            self.llm = None
            logger.warning("GeminiLLM not available, tagging will not work.")

    def generate_tag(self, text: str) -> str:
        if not self.llm:
            return "general_info"
        prompt = (
            "Given the following insurance policy text, assign ONE tag from this list: "
            f"{', '.join(FIXED_TAGS)}.\nText: {text}\nTag: "
        )
        tag = self.llm.generate_tag(prompt)
        tag = tag.strip().lower()
        if tag not in FIXED_TAGS:
            tag = "general_info"
        return tag

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller, overlapping chunks and tags each chunk.
        """
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        # Tag each chunk and add to metadata
        for chunk in chunks:
            tag = self.generate_tag(chunk.page_content)
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['tag'] = tag
        return chunks

    def generate_embeddings(self, texts: List[str]):
        """
        Generates embeddings for a list of text strings.
        """
        logger.info(f"Generating embeddings for {len(texts)} text items...")
        try:
            embeddings = self.embeddings_model.embed_documents(texts)
            logger.info("Embeddings generated successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_for_query(self, query: str):
        """
        Generates an embedding for a single query string.
        """
        try:
            embedding = self.embeddings_model.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

# Example usage (for testing this module)
if __name__ == "__main__":
    text_processor = TextProcessor()

    sample_documents = [
        Document(page_content="This is a very long paragraph about insurance policies and their terms and conditions. It covers various aspects like age limits, policy duration, and types of medical procedures covered. For example, general surgeries are always included, but specialized procedures might have waiting periods. Always refer to Section A.1.2 for detailed exclusions."),
        Document(page_content="Another document discussing claims processing. Claims typically involve submitting a form, providing medical reports, and waiting for approval. The approval process considers the policy's active status and any pre-existing conditions. Payouts are made within 30 days of approval.")
    ]

    # Chunking test
    chunks = text_processor.chunk_documents(sample_documents)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")

    # Embedding test
    sample_texts = [chunk.page_content for chunk in chunks]
    try:
        sample_embeddings = text_processor.generate_embeddings(sample_texts[:2]) # Just first two for brevity
        print(f"\nGenerated {len(sample_embeddings)} embeddings. Dimension: {len(sample_embeddings[0])}")
    except Exception as e:
        print(f"Embedding test failed: {e}")