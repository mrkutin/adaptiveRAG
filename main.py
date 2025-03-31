import asyncio
import logging
from typing import TypedDict, Annotated, Sequence, Dict
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatState(MessagesState):
    """State for the chat workflow."""
    telegram_chat_id: int
    messages: Annotated[Sequence[HumanMessage | AIMessage], "messages"]

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

class TelegramBot:
    """Class to manage Telegram bot operations."""
    def __init__(self):
        # Log configuration parameters
        logger.info("Starting bot with configuration:")
        logger.info(f"Telegram Bot Token: ...{settings.telegram_bot_token[-8:]}")
        logger.info(f"Ollama Base URL: {settings.ollama_base_url}")
        logger.info(f"Ollama Model: {settings.ollama_model}")
        logger.info(f"Ollama Temperature: {settings.ollama_temperature}")
        logger.info(f"Ollama Timeout: {settings.ollama_timeout}")
        logger.info(f"Debug Mode: {settings.debug}")
        logger.info(f"Log Level: {settings.log_level}")
        
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.workflow = WorkflowGraph(self.bot)
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register message handlers."""
        self.dp.message.register(self.cmd_start, CommandStart())
        self.dp.message.register(self.handle_message)
    
    async def cmd_start(self, message: Message):
        """Handle /start command."""
        await message.answer("Hello! Send me any message and I'll help you!")
    
    async def handle_message(self, message: Message):
        """Handle incoming messages."""
        try:
            # Initialize state
            initial_state = ChatState(
                messages=[HumanMessage(content=message.text)],
                telegram_chat_id=message.chat.id
            )
            
            # Process through workflow
            await self.workflow.process(initial_state)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message.answer(f"Sorry, I encountered an error: {str(e)}")
    
    async def start(self):
        """Start the bot."""
        try:
            logger.info("Starting bot polling...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Bot polling failed: {str(e)}")
            raise
        finally:
            logger.info("Bot stopped")

async def main():
    bot = TelegramBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main()) 