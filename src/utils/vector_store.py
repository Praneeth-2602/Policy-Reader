# src/utils/vector_store.py
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model # Expects an initialized embedding model (GoogleGenerativeAIEmbeddings)
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """
        Initializes and returns the Astra DB Vector Store instance.
        """
        if not settings.ASTRA_DB_APPLICATION_TOKEN or not settings.ASTRA_DB_API_ENDPOINT:
            raise ValueError(
                "Astra DB API Endpoint and Application Token must be set in .env or config.py."
            )
        try:
            vector_store = AstraDBVectorStore(
                embedding=self.embedding_model,
                collection_name=settings.ASTRA_DB_COLLECTION_NAME,
                token=settings.ASTRA_DB_APPLICATION_TOKEN,
                api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
                namespace=settings.ASTRA_DB_KEYSPACE, # Will use default if None
            )
            logger.info(f"Connected to Astra DB collection: {settings.ASTRA_DB_COLLECTION_NAME}")
            return vector_store
        except Exception as e:
            logger.error(f"Error connecting to Astra DB: {e}", exc_info=True) # Added exc_info=True
            raise

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of LangChain Document objects to the vector store.
        Embeddings will be generated automatically by the vector store using the provided embedding_model.
        """
        logger.info(f"Adding {len(documents)} documents to Astra DB...")
        try:
            # AstraDBVectorStore.add_documents handles batching internally
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to Astra DB.")
        except Exception as e:
            logger.error(f"Error adding documents to Astra DB: {e}", exc_info=True) # Added exc_info=True
            raise

    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a similarity search in the vector store for the given query.
        """
        logger.info(f"Retrieving top {k} documents for query: '{query}'")
        try:
            # LangChain's AstraDBVectorStore handles query embedding internally
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents.")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents from Astra DB: {e}", exc_info=True) # Added exc_info=True
            raise

# Example Usage (for testing this module)
if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN are set in .env
    # before running this example.

    from src.utils.text_processing import TextProcessor
    from src.loaders.document_loader import DocumentProcessor
    import os
    import asyncio # Added for a small async sleep

    async def run_vector_store_example(): # Wrapped in an async function
        # First, make sure you have some dummy documents in your knowledge_base directory
        # (or create one as in document_loader.py example)
        if not os.path.exists(settings.KNOWLEDGE_BASE_DIR):
            os.makedirs(settings.KNOWLEDGE_BASE_DIR)
            with open(os.path.join(settings.KNOWLEDGE_BASE_DIR, "policy_example.txt"), "w") as f:
                f.write("This is a sample policy document for a general insurance plan.\n")
                f.write("Section 1: Eligibility. Individuals aged 18-60 are eligible. Policy duration must be at least 6 months.\n")
                f.write("Section 2: Medical Procedures. Knee surgery is covered after a 3-month waiting period from policy inception. Max payout: 50,000 INR.\n")
                f.write("Section 3: Geographic Scope. Coverage applies to surgeries performed in metropolitan areas like Pune, Mumbai, Delhi, and Bangalore.\n")
                f.write("Section 4: Exclusions. Pre-existing conditions are not covered for the first 12 months.\n")
                f.write("Section 5: Claim Process. Submit all medical bills and a claim form within 30 days of discharge.")

        # 1. Initialize text processor for embeddings
        text_processor = TextProcessor()
        # The VectorStoreManager will use text_processor.embeddings_model for its embedding needs

        # 2. Initialize Vector Store Manager
        try:
            vector_store_manager = VectorStoreManager(text_processor.embeddings_model)

            # 3. Load documents (assuming document_loader.py is working)
            doc_processor = DocumentProcessor()
            loaded_documents = doc_processor.load_directory(settings.KNOWLEDGE_BASE_DIR)
            logger.info(f"Document loading completed. Loaded {len(loaded_documents)} documents.") # <-- New Log

            # 4. Add chunks to Astra DB
            # For explicit control, let's pass chunks generated by our TextProcessor.
            chunked_documents = text_processor.chunk_documents(loaded_documents)
            logger.info(f"Documents chunked. Created {len(chunked_documents)} chunks.") # <-- New Log

            vector_store_manager.add_documents(chunked_documents)
            logger.info("Documents successfully added to Vector Store Manager.") # <-- New Log

            # 5. Test retrieval
            query = "Can a 46-year-old male get coverage for knee surgery in Pune with a 3-month-old policy?"
            retrieved_docs = vector_store_manager.retrieve_relevant_documents(query, k=3)

            print(f"\n--- Retrieved Documents for query: '{query}' ---")
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')})")
                    print(f"Content: {doc.page_content.strip()}")
                    print(f"Metadata: {doc.metadata}")
            else:
                print("No relevant documents found in retrieval test.") # Changed wording
            
            await asyncio.sleep(1) # Small sleep to ensure logs flush if running quickly

        except ValueError as e:
            logger.error(f"Configuration Error: {e}. Please check your .env file and Astra DB setup.", exc_info=True)
            print(f"Configuration Error: {e}. Please check your .env file and Astra DB setup.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Astra DB test: {e}", exc_info=True)
            print(f"An unexpected error occurred during Astra DB test: {e}")

if __name__ == "__main__":
    asyncio.run(run_vector_store_example()) # Run the async example