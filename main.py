import asyncio
import logging
from typing import TypedDict, Annotated, Sequence
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

# Define our state
class BotState(MessagesState):
    telegram_chat_id: int

# Define our nodes
async def answer_node(state: BotState) -> BotState:
    """Generate and stream the response."""
    try:
         # Send initial "processing" message
        processing_msg = await bot.send_message(
            chat_id=state["telegram_chat_id"],
            text="Processing your request..."
        )

        # Get the last user message from history
        last_message = state["messages"][-1].content
        current_sentence = ""
        full_response = ""
        
        # Stream the response
        async for chunk in chain.astream({"input": last_message}):
            if chunk:
                current_sentence += chunk
                
                # Check if we have a complete sentence
                if any(current_sentence.strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                    full_response += current_sentence
                    current_sentence = ""
                    # Update the message with the accumulated response
                    await bot.edit_message_text(
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
        logger.error(f"Error in answer_node: {str(e)}")
        raise



# Create the graph
workflow = StateGraph(BotState)

# Add nodes
workflow.add_node("answer", answer_node)

# Add edges
workflow.add_edge(START, "answer")
workflow.add_edge("answer", END)


# Compile the graph
app = workflow.compile()

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Hello! Send me any message and I'll help you!")

@dp.message()
async def handle_message(message: Message):
    try:
        # Initialize state
        initial_state = BotState(
            messages=[HumanMessage(content=message.text)],
            telegram_chat_id=message.chat.id
        )
        
        # Run the graph
        await app.ainvoke(initial_state)
        
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