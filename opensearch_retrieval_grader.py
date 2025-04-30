import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config import settings
from typing import Literal
logger = logging.getLogger(__name__)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class OpenSearchRetrievalGrader:
    """Class to handle document relevance grading."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["question", "document"],
            template="""
            You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Retrieved document: \n\n {document} \n\n User question: {question}
            """
        )

        self._llm = ChatOllama(
            base_url=settings.retrieval_grader_ollama_base_url,
            model=settings.retrieval_grader_ollama_model,
            temperature=settings.retrieval_grader_ollama_temperature,
            streaming=True,
            num_ctx=settings.retrieval_grader_ollama_num_ctx
        )

        structured_llm = self._llm.with_structured_output(GradeDocuments)
        self.chain = self._prompt | structured_llm

    def invoke(self, question: str, document: str) -> str:
        """Grade document relevance to the question."""
        try:
            result = self.chain.invoke({
                "question": question,
                "document": document
            })
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in RetrievalGrader.invoke: {str(e)}")
            raise 

    async def ainvoke(self, question: str, document: str) -> str:
        """Grade document relevance to the question."""
        try:
            result = await self.chain.ainvoke({
                "question": question,
                "document": document
            })  
            return result.binary_score
        except Exception as e:
            logger.error(f"Error in RetrievalGrader.ainvoke: {str(e)}")
            raise 
