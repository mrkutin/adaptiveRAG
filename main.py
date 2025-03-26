import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log configuration parameters
logger.info("Starting bot with configuration:")
logger.info(f"Telegram Bot Token: ...{settings.telegram_bot_token[-8:]}")
logger.info(f"Ollama Base URL: {settings.ollama_base_url}")
logger.info(f"Ollama Model: {settings.ollama_model}")
logger.info(f"Ollama Temperature: {settings.ollama_temperature}")
logger.info(f"Ollama Timeout: {settings.ollama_timeout}")
logger.info(f"Debug Mode: {settings.debug}")
logger.info(f"Log Level: {settings.log_level}")

# Initialize bot and dispatcher
bot = Bot(token=settings.telegram_bot_token)
dp = Dispatcher()

# Initialize Ollama with streaming enabled
llm = OllamaLLM(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=settings.ollama_temperature,
    timeout=settings.ollama_timeout,
    streaming=True
)

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Provide clear and concise answers."),
    ("human", "{input}")
])

# Create the chain with streaming
chain = prompt | llm | StrOutputParser()

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Hello! Send me any message and I'll help you!")

@dp.message()
async def handle_message(message: Message):
    try:
        # Send initial "processing" message
        processing_msg = await message.answer("Processing your request...")
        full_response = ""
        current_sentence = ""
        
        # Stream the response
        async for chunk in chain.astream({"input": message.text}):
            if chunk:
                current_sentence += chunk
                
                # Check if we have a complete sentence (ends with punctuation)
                if any(current_sentence.strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                    full_response += current_sentence
                    current_sentence = ""
                    # Update the message with the accumulated response
                    await processing_msg.edit_text(full_response + "▌")
        
        # Add any remaining text
        if current_sentence:
            full_response += current_sentence
        
        # Final update without the cursor
        await processing_msg.edit_text(full_response)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await message.answer(f"Sorry, I encountered an error: {str(e)}")

async def main():
    try:
        # Start polling
        logger.info("Starting bot polling...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Bot polling failed: {str(e)}")
        raise
    finally:
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main()) 