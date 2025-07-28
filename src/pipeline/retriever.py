# src/pipeline/retriever.py
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.vector_store import VectorStoreManager
from src.agents.models import ParsedQuery
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseRetriever:
    def __init__(self):
        # Initialize the embedding model (needed by VectorStoreManager)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY
        )
        # Initialize the VectorStoreManager with the embedding model
        self.vector_store_manager = VectorStoreManager(self.embedding_model)
        logger.info("ClauseRetriever initialized with Astra DB Vector Store.")

    def retrieve_clauses(self, parsed_query: ParsedQuery, k: int = 5) -> List[Document]:
        """
        Retrieves relevant clauses/documents from the vector store based on the parsed query.
        The query sent to the vector store is a synthesized string from the parsed query.
        """
        # Synthesize a search query string from the parsed query object
        search_query_parts = []
        if parsed_query.age:
            search_query_parts.append(f"{parsed_query.age}-year-old")
        if parsed_query.gender:
            search_query_parts.append(parsed_query.gender)
        if parsed_query.procedure:
            search_query_parts.append(parsed_query.procedure)
        if parsed_query.location:
            search_query_parts.append(parsed_query.location)
        if parsed_query.policy_duration_months:
            search_query_parts.append(f"{parsed_query.policy_duration_months}-month policy")
        if parsed_query.keywords:
            search_query_parts.extend(parsed_query.keywords)

        # If no specific parts found, use a generic query or raise error
        if not search_query_parts:
            logger.warning("Parsed query is too vague, using generic keywords for retrieval.")
            synthesized_query = "insurance policy coverage"
        else:
            synthesized_query = ", ".join(search_query_parts) + " insurance policy rules"

        logger.info(f"Synthesized query for retrieval: '{synthesized_query}'")

        # Perform the retrieval from Astra DB
        retrieved_documents = self.vector_store_manager.retrieve_relevant_documents(synthesized_query, k=k)

        if not retrieved_documents:
            logger.warning(f"No documents retrieved for query: '{synthesized_query}'")
        return retrieved_documents

# Example Usage (for testing this module)
if __name__ == "__main__":
    import asyncio
    # Ensure GOOGLE_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN are set in .env
    # And that documents have been loaded into your Astra DB vector store (from previous step)

    async def main():
        # First, ensure your Astra DB has documents loaded.
        # You might need to run src/utils/vector_store.py's example part once to populate the DB.
        print("Please ensure your Astra DB has documents loaded before running this example.")
        print("Run `python src/utils/vector_store.py` first if you haven't.")
        await asyncio.sleep(2) # Give user time to read

        # Initialize the retriever
        clause_retriever = ClauseRetriever()

        # Create a dummy ParsedQuery object (this would normally come from the QueryParser)
        sample_parsed_query = ParsedQuery(
            age=46,
            gender="male",
            procedure="knee surgery",
            location="Pune",
            policy_duration_months=3,
            keywords=["coverage", "new policy"]
        )

        try:
            retrieved_clauses = await clause_retriever.retrieve_clauses(sample_parsed_query, k=3)

            print(f"\n--- Retrieved Clauses for Parsed Query: {sample_parsed_query.json(indent=2)} ---")
            if retrieved_clauses:
                for i, doc in enumerate(retrieved_clauses):
                    print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')})")
                    print(f"Content: {doc.page_content.strip()}")
                    print(f"Metadata: {doc.metadata}")
            else:
                print("No relevant clauses found.")

        except Exception as e:
            print(f"\n--- Error during retrieval test ---")
            print(f"Error: {e}")

    asyncio.run(main())