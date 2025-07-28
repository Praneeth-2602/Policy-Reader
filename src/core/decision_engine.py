# src/core/decision_engine.py
from typing import List
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from config import settings
from src.agents.models import ParsedQuery, DecisionResponse, Justification # Import new models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.2, # Slightly higher temperature for reasoning, but still controlled
            google_api_key=settings.GEMINI_API_KEY,
        )
        self.output_parser = PydanticOutputParser(pydantic_object=DecisionResponse)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Prompt template for the decision engine
        # Prompt template for the decision engine (concise, direct answer only)
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert insurance policy evaluator. Given a user's question and relevant policy clauses, answer the question as directly and concisely as possible, using only the provided context. Do NOT use external knowledge. If the answer is not present in the context, reply with 'Information not found in the provided policy clauses.' Do not include explanations, justifications, or extra formatting. Only return the direct answer as a single sentence."
            ),
            HumanMessagePromptTemplate.from_template(
                "User Query: {parsed_query}\n\nCONTEXTUAL POLICY CLAUSES:\n{context}\n\nFormat instructions:\n{format_instructions}"
            ),
        ])

        self.decision_chain = self.prompt_template | self.llm | self.output_parser
        logger.info("DecisionEngine initialized.")

    async def evaluate_claim(self, parsed_query: ParsedQuery, retrieved_documents: List[Document]) -> DecisionResponse:
        """
        Evaluates an insurance claim based on parsed query details and retrieved policy clauses.
        """
        logger.info("Evaluating claim with parsed query and retrieved documents.")

        # Prepare context for the LLM
        context_parts = []
        for i, doc in enumerate(retrieved_documents):
            source_ref = doc.metadata.get('source', 'N/A')
            page_ref = doc.metadata.get('page_number', 'N/A')
            context_parts.append(
                f"--- Clause {i+1} (Source: {source_ref}, Page: {page_ref}) ---\n"
                f"{doc.page_content.strip()}\n"
            )
        context_str = "\n\n".join(context_parts)

        if not context_str:
            logger.warning("No context provided to DecisionEngine. Returning 'requires more information'.")
            return DecisionResponse(
                decision="requires more information",
                amount=None,
                justification=[
                    Justification(
                        clause_snippet="No relevant policy clauses found.",
                        clause_reference="N/A",
                        reasoning="The system could not retrieve any relevant clauses from the knowledge base to make a decision. Please provide more specific details or relevant policy documents."
                    )
                ]
            )

        try:
            # Invoke the chain asynchronously
            result = await self.decision_chain.ainvoke({
                "parsed_query": parsed_query.model_dump_json(), # Use model_dump_json for Pydantic V2 compatibility
                "context": context_str,
                "format_instructions": self.format_instructions
            })
            logger.info(f"Decision Engine output: {result.dict()}")
            return result
        except Exception as e:
            logger.error(f"Error in DecisionEngine during claim evaluation: {e}", exc_info=True)
            # Handle potential parsing errors or LLM failures gracefully
            return DecisionResponse(
                decision="error",
                amount=None,
                justification=[
                    Justification(
                        clause_snippet="System error during decision evaluation.",
                        clause_reference="N/A",
                        reasoning=f"An internal error occurred: {str(e)}. Please try again or refine the query."
                    )
                ]
            )

# Example Usage (for testing this module)
if __name__ == "__main__":
    import asyncio
    # Ensure GOOGLE_API_KEY is set in .env.
    # Also, ensure your Astra DB has documents loaded via src/utils/vector_store.py
    # and that src/agents/query_parser.py works correctly.

    async def main():
        decision_engine = DecisionEngine()

        # Create a dummy parsed query (from query_parser.py's output)
        dummy_parsed_query = ParsedQuery(
            age=46,
            gender="male",
            procedure="knee surgery",
            location="Pune",
            policy_duration_months=3,
            keywords=["coverage", "new policy"]
        )

        # Create dummy retrieved documents (from src/pipeline/retriever.py's output)
        dummy_retrieved_documents = [
            Document(
                page_content="Section 2: Medical Procedures. Knee surgery is covered after a 3-month waiting period from policy inception. Max payout: 50,000 INR.",
                metadata={"source": "Policy_Doc_1.pdf", "page_number": 5, "category": "UncategorizedText"}
            ),
            Document(
                page_content="Section 3: Geographic Scope. Coverage applies to surgeries performed in metropolitan areas like Pune, Mumbai, Delhi, and Bangalore.",
                metadata={"source": "Policy_Doc_1.pdf", "page_number": 6, "category": "UncategorizedText"}
            ),
            Document(
                page_content="Section 1: Eligibility. Individuals aged 18-60 are eligible.",
                metadata={"source": "Policy_Doc_1.pdf", "page_number": 2, "category": "UncategorizedText"}
            ),
            # Add a conflicting/irrelevant doc to test LLM's focus
            Document(
                page_content="Section 7: Dental Coverage. Root canal procedures are covered.",
                metadata={"source": "Policy_Doc_1.pdf", "page_number": 10, "category": "UncategorizedText"}
            )
        ]

        try:
            decision_result = await decision_engine.evaluate_claim(
                dummy_parsed_query, dummy_retrieved_documents
            )
            print(f"\n--- Final Decision for Sample Scenario ---")
            print(decision_result.json(indent=2))

        except Exception as e:
            print(f"\n--- Error during Decision Engine test ---")
            print(f"Error: {e}")

    asyncio.run(main())