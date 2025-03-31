import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END

from config import settings

logger = logging.getLogger(__name__)

class ChatState(MessagesState):
    """State for the chat workflow."""
    telegram_chat_id: int
    messages: Annotated[Sequence[HumanMessage | AIMessage], "messages"]
    documents: Annotated[Sequence[Document], "documents"]

class ChatChain:
    """Class to handle chat processing logic."""
    def __init__(self, bot: Bot):
        self.bot = bot
        self.llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            timeout=settings.ollama_timeout,
            streaming=True,
            max_tokens=settings.ollama_max_tokens
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear and concise answers."),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def process_message(self, state: ChatState) -> ChatState:
        """Process a message and update the state."""
        try:
            # Send initial "processing" message
            processing_msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Processing your request..."
            )

            # Get the last user message from history
            last_message = state["messages"][-1].content
            current_sentence = ""
            full_response = ""
            
            # Stream the response
            async for chunk in self.chain.astream({"input": last_message}):
                if chunk:
                    current_sentence += chunk
                    
                    # Check if we have a complete sentence
                    if any(current_sentence.strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                        full_response += current_sentence
                        current_sentence = ""
                        # Update the message with the accumulated response
                        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=processing_msg.message_id,
                            text=full_response
                        )
            
            # Add any remaining text
            if current_sentence:
                full_response += current_sentence
            
            # Add the AI response to message history
            state["messages"].append(AIMessage(content=full_response))
            
            return state
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            raise

class WorkflowGraph:
    """Class to manage the workflow graph."""
    def __init__(self, bot: Bot):
        self.chat_chain = ChatChain(bot)
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("answer", self.chat_chain.process_message)
        
        # Add edges
        self.workflow.add_edge(START, "answer")
        self.workflow.add_edge("answer", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 