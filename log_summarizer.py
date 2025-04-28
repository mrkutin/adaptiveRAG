import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from config import settings
from typing import List
from langchain_core.documents import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

logger = logging.getLogger(__name__)

class LogSummary(BaseModel):
    """Summary of logs with key information."""
    summary: str = Field(description="A concise summary of the logs")
    key_events: List[str] = Field(description="List of key events or patterns found in the logs")
    error_count: int = Field(description="Number of error events found")
    warning_count: int = Field(description="Number of warning events found")
    stack_traces: List[str] = Field(description="List of stack traces found in the logs")

class LogSummarizer:
    """Class to handle log summarization."""
    def __init__(self):
        # Define the prompt for summarizing logs
        self._prompt = PromptTemplate(
            input_variables=["logs"],
            template="""
            You are a log analysis expert. Analyze the following logs and provide a summary.
            Focus on:
            1. Key events and patterns
            2. Error and warning counts
            3. Any unusual or important occurrences
            4. Stack traces (preserve them exactly as they appear)
            
            Logs to analyze:
            {logs}
            
            Provide a structured summary that includes:
            - A concise overview of what happened
            - List of key events or patterns
            - Count of errors and warnings
            - List of stack traces (if any)
            """
        )

        # Initialize the LLM
        self._llm = ChatOllama(
            base_url=settings.log_summarizer_ollama_base_url,
            model=settings.log_summarizer_ollama_model,
            temperature=settings.log_summarizer_ollama_temperature,
            timeout=settings.log_summarizer_ollama_timeout,
            max_tokens=settings.log_summarizer_ollama_max_tokens
        )

        # Create structured output chain
        structured_llm = self._llm.with_structured_output(LogSummary)
        self.chain = self._prompt | structured_llm

    def invoke(self, logs: List[str]) -> LogSummary:
        """Summarize a list of logs.
        
        Args:
            logs: List of log entries to summarize
            
        Returns:
            LogSummary object containing the summary and analysis
        """
        try:
            # Combine logs into a single string
            combined_logs = "\n".join(logs)
            
            # Get summary
            result = self.chain.invoke({
                "logs": combined_logs
            })
            
            # Deduplicate stack traces
            if result.stack_traces:
                result.stack_traces = list(dict.fromkeys(result.stack_traces))
            
            return result
        except Exception as e:
            logger.error(f"Error in LogSummarizer.invoke: {str(e)}")
            raise

    async def ainvoke(self, logs: List[str]) -> LogSummary:
        """Summarize a list of logs asynchronously.
        
        Args:
            logs: List of log entries to summarize
            
        Returns:
            LogSummary object containing the summary and analysis
        """
        try:
            # Combine logs into a single string
            combined_logs = "\n".join(logs)
            
            # Get summary
            result = await self.chain.ainvoke({
                "logs": combined_logs
            })
            
            # Deduplicate stack traces
            if result.stack_traces:
                result.stack_traces = list(dict.fromkeys(result.stack_traces))
            
            return result
        except Exception as e:
            logger.error(f"Error in LogSummarizer.ainvoke: {str(e)}")
            raise 