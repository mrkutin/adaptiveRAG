import logging
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
import os
from datetime import datetime
import asyncio
import re

logger = logging.getLogger(__name__)

class Answerer:
    """Class to handle answer generation."""
    def __init__(self):
        self._prompt = PromptTemplate(
            input_variables=["question", "context", "stack_trace", "code_context"],
            template="""
Based on the user's query and the logs provided, determine the appropriate response.

USER QUESTION: {question}

LOGS CONTEXT:
{context}

STACK TRACE:
{stack_trace}

CODE CONTEXT:
{code_context}

INSTRUCTIONS:
Please provide:
1. A direct answer to the user's question, if possible (e.g., confirmation of an event or status). If the logs contain information that directly answers the user's question, state it clearly.
2. A concise description of what these logs represent, suitable for a business user.
3. Technical context from the codebase, if applicable (relevant files, functions, or code paths).
4. Exact IDs affected by the error, if applicable. Pay special attention to:
   - Order IDs (recid, order_number)
   - MongoDB IDs (_id)
   - Contract numbers
   - Invoice numbers
   - Product record IDs (rec_id)

Focus on providing a clear and direct response to the user's question, supplemented by technical insights when necessary. Make sure to explicitly list all relevant identifiers from the error payload.
"""
        )

        self._llm = ChatOllama(
            base_url=settings.answerer_ollama_base_url,
            model=settings.answerer_ollama_model,
            temperature=settings.answerer_ollama_temperature,
            num_ctx=settings.answerer_ollama_num_ctx
        )

        self.chain = self._prompt | self._llm | StrOutputParser()

    def _debug_prompt(self, inputs: dict) -> None:
        """Debug the prompt by showing the complete template with injected values and saving to file."""
        try:
            complete_prompt = self._prompt.format(**inputs)
            
            # Create debug directory if it doesn't exist
            debug_dir = "debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/debug_prompt_{timestamp}.txt"
            
            # Write to file
            with open(filename, 'w') as f:
                f.write(complete_prompt)
            
            logger.debug(f"Debug prompt written to {filename}")
            logger.debug("\n=== DEBUG: COMPLETE PROMPT ===\n%s\n=== END DEBUG ===", complete_prompt)
        except Exception as e:
            logger.error(f"Error while debugging prompt: {str(e)}")

    def astream(self, inputs: dict):
        """Stream the answer generation."""
        self._debug_prompt(inputs)
        return self.chain.astream(inputs)
    
    def ainvoke(self, inputs: dict):
        """Invoke the answer generation."""
        self._debug_prompt(inputs)
        return self.chain.ainvoke(inputs)

def parse_debug_file(content: str) -> tuple[str, str, str, str]:
    """Parse the debug file content to extract all components.
    
    Returns:
        tuple: (question, context, stack_trace, code_context)
    """
    # Find the question
    question_match = re.search(r"USER QUESTION: (.*?)\n\s*LOGS CONTEXT:", content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # Find the context
    context_match = re.search(r"LOGS CONTEXT:\s*(.*?)\n\s*STACK TRACE:", content, re.DOTALL)
    context = context_match.group(1).strip() if context_match else ""

    # Find the stack trace
    stack_trace_match = re.search(r"STACK TRACE:\s*(.*?)\n\s*CODE CONTEXT:", content, re.DOTALL)
    stack_trace = stack_trace_match.group(1).strip() if stack_trace_match else ""

    # Find the code context
    code_context_match = re.search(r"CODE CONTEXT:\s*(.*?)\n\s*INSTRUCTIONS:", content, re.DOTALL)
    code_context = code_context_match.group(1).strip() if code_context_match else ""

    return question, context, stack_trace, code_context

async def main():
    """Main function to load and run the most recent debug prompt."""
    try:
        # Find the most recent debug prompt file
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            print("No debug directory found.")
            return

        debug_files = [f for f in os.listdir(debug_dir) if f.startswith('debug_prompt_')]
        if not debug_files:
            print("No debug prompt files found.")
            return

        # Get the most recent file
        latest_file = max(debug_files, key=lambda x: os.path.getctime(os.path.join(debug_dir, x)))
        latest_path = os.path.join(debug_dir, latest_file)

        # Read the prompt
        with open(latest_path, 'r') as f:
            content = f.read()

        print(f"Loading prompt from: {latest_path}")
        print("\nPrompt content:")
        print(content)

        # Parse the content
        question, context, stack_trace, code_context = parse_debug_file(content)
        print("\nExtracted components:")
        print(f"Question: {question}")
        print(f"Context length: {len(context)} characters")
        print(f"Stack trace length: {len(stack_trace)} characters")
        print(f"Code context length: {len(code_context)} characters")
        print("\nGenerating answer...")

        # Initialize answerer and run the prompt
        answerer = Answerer()
        response = await answerer.ainvoke({
            "question": question,
            "context": context,
            "stack_trace": stack_trace,
            "code_context": code_context
        })

        print("\nGenerated response:")
        print(response)

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())