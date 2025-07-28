# test_document_processing.py
import os
from src.loaders.document_loader import DocumentProcessor
from src.utils.text_processing import TextProcessor
from config import settings
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_test():
    logger.info("Starting document processing test.")

    # Ensure dummy file exists
    if not os.path.exists(settings.KNOWLEDGE_BASE_DIR):
        os.makedirs(settings.KNOWLEDGE_BASE_DIR)
    
    test_file_path = os.path.join(settings.KNOWLEDGE_BASE_DIR, "policy_example.txt")
    if not os.path.exists(test_file_path):
        with open(test_file_path, "w") as f:
            f.write("This is a simple test policy. Section 1 covers general conditions. Section 2 covers specific procedures like knee surgery. This is a longer sentence to test chunking capabilities.")


    doc_processor = DocumentProcessor()
    loaded_documents = doc_processor.load_directory(settings.KNOWLEDGE_BASE_DIR)
    
    if not loaded_documents:
        logger.error("No documents were loaded by DocumentProcessor. Check file path and content.")
        return

    logger.info(f"Successfully loaded {len(loaded_documents)} documents.")
    for i, doc in enumerate(loaded_documents):
        logger.info(f"Loaded Doc {i+1}: Source={doc.metadata.get('source')}, Content='{doc.page_content[:100]}...'")

    text_processor = TextProcessor() # This initializes embedding model
    
    chunks = text_processor.chunk_documents(loaded_documents)
    
    if not chunks:
        logger.error("No chunks were created by TextProcessor. Check chunking logic or input documents.")
        return

    logger.info(f"Successfully created {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: Content='{chunk.page_content[:100]}...'")

    # Attempt to generate embeddings for a small subset of chunks
    try:
        sample_texts_for_embedding = [chunk.page_content for chunk in chunks[:min(3, len(chunks))]] # Take up to 3 chunks
        if sample_texts_for_embedding:
            logger.info(f"Attempting to generate embeddings for {len(sample_texts_for_embedding)} sample texts.")
            sample_embeddings = text_processor.generate_embeddings(sample_texts_for_embedding)
            logger.info(f"Generated {len(sample_embeddings)} sample embeddings. Dimension: {len(sample_embeddings[0])}")
        else:
            logger.warning("No chunks available to generate sample embeddings.")

    except Exception as e:
        logger.error(f"Error during embedding generation test: {e}", exc_info=True)
        # Check for quota errors, API key issues here
        if "quota" in str(e).lower() or "api key" in str(e).lower():
            logger.error("Possible Google API Key or Quota issue during embedding generation.")


    logger.info("Document processing test completed.")
    await asyncio.sleep(1) # Small sleep to allow logs to flush

if __name__ == "__main__":
    asyncio.run(run_test())