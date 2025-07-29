import os
from typing import List
from langchain_core.documents import Document
import logging
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        pass

    def load_document(self, file_path: str) -> List[Document]:
        """
        Loads a single document based on its file type.
        Uses PyMuPDF for .pdf files (text-based only), and returns Document objects.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        docs = []
        try:
            if file_extension == ".pdf":
                logger.info(f"Using PyMuPDF for text extraction from {file_path}")
                text = self.extract_text_from_pdf(file_path)
                if text.strip():
                    doc = Document(page_content=text, metadata={
                        "source": os.path.basename(file_path),
                        "page_number": 1
                    })
                    docs.append(doc)
                else:
                    logger.warning(f"No text extracted from {file_path}.")
            elif file_extension == ".txt":
                logger.info(f"Using TextLoader for {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc = Document(page_content=text, metadata={
                    "source": os.path.basename(file_path),
                    "page_number": 1
                })
                docs.append(doc)
            else:
                logger.warning(f"Unsupported file type: {file_extension}. Skipping {file_path}.")
            return docs
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}", exc_info=True)
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

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
        logger.info(f"Finished processing directory: {directory_path}. Loaded {len(all_docs)} document pages/elements.") 
        return all_docs

# Example usage (for testing this module)
if __name__ == "__main__":
    pdf_path = "data/knowledge_base/policy_042f627c.pdf"
    processor = DocumentProcessor()
    docs = processor.load_document(pdf_path)
    for doc in docs:
        print(doc.page_content)