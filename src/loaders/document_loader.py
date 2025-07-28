# src/loaders/document_loader.py
import os
from typing import List
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
    # DirectoryLoader, # Not directly used in load_document, but fine to keep
    TextLoader # <-- ADD THIS IMPORT
)
from langchain_core.documents import Document
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        pass

    def load_document(self, file_path: str) -> List[Document]:
        """
        Loads a single document based on its file type.
        Uses TextLoader for .txt files and Unstructured loaders for others.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        loader = None

        try:
            if file_extension == ".txt":
                loader = TextLoader(file_path)
                logger.info(f"Using TextLoader for {file_path}")
            elif file_extension == ".pdf":
                # UnstructuredPDFLoader may not accept 'mode' and 'strategy' in all versions
                loader = UnstructuredPDFLoader(file_path)
                logger.info(f"Using UnstructuredPDFLoader for {file_path}")
            elif file_extension in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
                logger.info(f"Using UnstructuredWordDocumentLoader for {file_path}")
            elif file_extension in [".eml", ".msg"]:
                loader = UnstructuredEmailLoader(file_path)
                logger.info(f"Using UnstructuredEmailLoader for {file_path}")
            else:
                logger.warning(f"Unsupported file type: {file_extension}. Attempting generic UnstructuredFileLoader.")
                loader = UnstructuredFileLoader(file_path)

            docs = loader.load() if loader else []
            # Add source metadata to each document
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                if "page_number" not in doc.metadata:
                    doc.metadata["page_number"] = 1
            return docs
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}", exc_info=True)
            return []

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Loads all supported documents from a given directory.
        """
        logger.info(f"Loading documents from directory: {directory_path}")
        all_docs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_docs.extend(self.load_document(file_path))
        # This log will only print if the loop completes successfully
        logger.info(f"Finished processing directory: {directory_path}. Loaded {len(all_docs)} document pages/elements.") 
        return all_docs

# Example usage (for testing this module)
if __name__ == "__main__":
    # This section is for direct testing and won't be executed when imported
    # It's good practice to keep these as separate test files or functions
    # For now, let's keep the core logic clean and assume its direct test is elsewhere.
    pass