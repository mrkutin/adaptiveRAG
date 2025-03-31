import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from langchain_core.messages import HumanMessage

from config import settings
from workflow import ChatState, WorkflowGraph

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                telegram_chat_id=message.chat.id,
                documents=[]  # Initialize empty documents list
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