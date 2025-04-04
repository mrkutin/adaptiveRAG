import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config import settings

logger = logging.getLogger(__name__)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class HallucinationGrader:
    """Class to handle hallucination grading."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["generation", "documents"],
            template="""
            You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
            Set of facts: \n\n {documents} \n\n LLM generation: {generation}
            """
        )

        self._llm = ChatOllama(
            base_url=settings.hallucination_grader_ollama_base_url,
            model=settings.hallucination_grader_ollama_model,
            temperature=settings.hallucination_grader_ollama_temperature,
            timeout=settings.hallucination_grader_ollama_timeout,
            streaming=True,
            max_tokens=settings.hallucination_grader_ollama_max_tokens
        )

        structured_llm = self._llm.with_structured_output(GradeHallucinations)
        self.chain = self._prompt | structured_llm

    def invoke(self, generation: str, documents: str) -> str:
        """Grade hallucinations in the generation."""
        try:
            result = self.chain.invoke({"documents": documents, "generation": generation})
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in HallucinationGrader.invoke: {str(e)}")
            raise 

    async def ainvoke(self, generation: str, documents: str) -> str:
        """Grade hallucinations in the generation."""
        try:
            result = await self.chain.ainvoke({
                "generation": generation,
                "documents": documents
            })  
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in HallucinationGrader.ainvoke: {str(e)}")
            raise 
