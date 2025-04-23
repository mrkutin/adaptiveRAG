import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from config import settings

logger = logging.getLogger(__name__)

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class AnswerGrader:
    """Class to handle answer grading."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["question", "generation"],
            template="""
            You are a grader assessing whether an answer addresses / resolves a question
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
            User question: \n\n {question} \n\n LLM generation: {generation}
            """
        )

        self._llm = ChatOllama(
            base_url=settings.answer_grader_ollama_base_url,
            model=settings.answer_grader_ollama_model,
            temperature=settings.answer_grader_ollama_temperature,
            timeout=settings.answer_grader_ollama_timeout,
            streaming=True,
            max_tokens=settings.answer_grader_ollama_max_tokens
        )

        structured_llm = self._llm.with_structured_output(GradeAnswer)
        self.chain = self._prompt | structured_llm

    def invoke(self, question: str, generation: str) -> str:
        """Grade answer."""
        try:
            result = self.chain.invoke({
                "question": question, 
                "generation": generation
            })
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in AnswerGrader.invoke: {str(e)}")
            raise 

    async def ainvoke(self, question: str, generation: str) -> str:
        """Grade answer."""
        try:
            result = await self.chain.ainvoke({
                "question": question,
                "generation": generation
            })  
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in AnswerGrader.ainvoke: {str(e)}")
            raise 
