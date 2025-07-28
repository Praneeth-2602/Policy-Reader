# src/agents/query_parser.py
from typing import Type
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from config import settings
from src.agents.models import ParsedQuery # Import our Pydantic model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryParser:
    def __init__(self, output_schema: Type[BaseModel] = ParsedQuery):
        """
        Initializes the QueryParser with a Gemini LLM and Pydantic output parser.
        Args:
            output_schema: The Pydantic model to parse the LLM output into.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0, # Low temperature for consistent structured output
            google_api_key=settings.GEMINI_API_KEY,
            # Enable JSON mode for direct structured output if model supports it
            # Note: For Gemini 1.5 Pro, setting response_mime_type is preferred for JSON mode.
            # Some LangChain versions might handle this automatically with PydanticOutputParser.
            # If explicit JSON mode is needed:
            # client_options={"model_name": settings.LLM_MODEL_NAME, "response_mime_type": "application/json"}
        )
        self.output_parser = PydanticOutputParser(pydantic_object=output_schema)

        # Get format instructions from the parser
        format_instructions = self.output_parser.get_format_instructions()

        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert query parser. Your task is to extract key details "
                "from a natural language insurance query and convert them into a structured JSON format. "
                "Be precise and fill only the fields that are directly mentioned or clearly inferable. "
                "If a field is not present or inferable, leave it as null/None. "
                "Do not make up information. Ensure the output strictly adheres to the provided JSON schema."
            ),
            HumanMessagePromptTemplate.from_template(
                "Parse the following insurance query into a JSON object:\n\n"
                "Query: {query}\n\n"
                "Format instructions:\n{format_instructions}"
            ),
        ])

        # Create the parsing chain using LangChain Expression Language (LCEL)
        self.parsing_chain = self.prompt_template | self.llm | self.output_parser
        logger.info("QueryParser initialized.")

    async def parse_query(self, query: str) -> ParsedQuery:
        """
        Parses a natural language query into a structured ParsedQuery object.
        """
        logger.info(f"Attempting to parse query: '{query}'")
        try:
            # Invoke the chain asynchronously
            parsed_data = await self.parsing_chain.ainvoke({"query": query, "format_instructions": self.output_parser.get_format_instructions()})
            logger.info(f"Successfully parsed query. Result: {parsed_data.dict()}")
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}", exc_info=True)
            # Depending on robustness needed, you might return a partially filled object
            # or raise a specific error. For a hackathon, raising might be acceptable.
            raise

# Example Usage (for testing this module)
if __name__ == "__main__":
    import asyncio
    # Ensure GOOGLE_API_KEY is set in .env before running this example.

    async def main():
        query_parser = QueryParser()

        sample_queries = [
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "Female, 28, got a dental procedure in Mumbai, policy just started last month",
            "Heart surgery, 50 years old, existing policy for 2 years",
            "Looking for info on eye treatment, 35M",
            "My policy covers accidents for my son (10M)",
            "What's covered for general check-ups?" # Test vague/incomplete
        ]

        for query_text in sample_queries:
            try:
                parsed_result = await query_parser.parse_query(query_text)
                print(f"\n--- Original Query: '{query_text}' ---")
                print("Parsed Result (JSON):")
                print(parsed_result.json(indent=2))
            except Exception as e:
                print(f"\n--- Error processing query: '{query_text}' ---")
                print(f"Error: {e}")

    asyncio.run(main())