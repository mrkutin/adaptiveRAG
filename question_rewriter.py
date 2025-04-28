import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from config import settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RewriteQuestion(BaseModel):
    """Improved question."""
    improved_question: str = Field(
        description="Improved Question"
    )

class QuestionRewriter:
    """Class to rewrite question."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You a question re-writer that converts an input question to a better version that is optimized 
            for opensearch retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."
            
            Here is the initial question: \n\n {question} \n\n Formulate an improved question.
            """
        )

        self._llm = ChatOllama(
            base_url=settings.question_rewriter_ollama_base_url,
            model=settings.question_rewriter_ollama_model,
            temperature=settings.question_rewriter_ollama_temperature,
            timeout=settings.question_rewriter_ollama_timeout,
            streaming=True,
            max_tokens=settings.question_rewriter_ollama_max_tokens
        )

        structured_llm = self._llm.with_structured_output(RewriteQuestion)
        self.chain = self._prompt | structured_llm
    
    def invoke(self, inputs: dict) -> str:
        """Rewrite the question."""
        try:
            result = self.chain.invoke(inputs)
            return result.improved_question
        except Exception as e:
            logger.error(f"Error in QuestionRewriter.invoke: {str(e)}")
            raise 

    async def ainvoke(self, inputs: dict) -> str:
        """Rewrite the question asynchronously."""
        try:
            result = await self.chain.ainvoke(inputs)
            return result.improved_question
        except Exception as e:
            logger.error(f"Error in QuestionRewriter.ainvoke: {str(e)}")
            raise