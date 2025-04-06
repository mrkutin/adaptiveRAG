import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings

logger = logging.getLogger(__name__)

class Answerer:
    """Class to handle answer generation."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Based on the user's query and the logs provided, determine the appropriate response.

            USER QUESTION: {question}

            LOGS CONTEXT:
            {context}

            Please provide:
            1. A direct answer to the user's question, if possible (e.g., confirmation of an event or status). If the logs contain information that directly answers the user's question, state it clearly.
            2. A concise description of what these logs represent, suitable for a business user.
            3. Technical context from the codebase, if applicable (relevant files, functions, or code paths).
            4. Exact IDs affected by the error, if applicable.

            Focus on providing a clear and direct response to the user's question, supplemented by technical insights when necessary.
            """
        )

        self._llm = ChatOllama(
            base_url=settings.answerer_ollama_base_url,
            model=settings.answerer_ollama_model,
            temperature=settings.answerer_ollama_temperature,
            timeout=settings.answerer_ollama_timeout,
            max_tokens=settings.answerer_ollama_max_tokens
        )

        self.chain = self._prompt | self._llm | StrOutputParser()

    def astream(self, inputs: dict):
        """Stream the answer generation."""
        return self.chain.astream(inputs)
    
    def ainvoke(self, inputs: dict):
        """Invoke the answer generation."""
        return self.chain.ainvoke(inputs)