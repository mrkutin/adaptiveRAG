# Adaptive RAG Telegram Bot

A Telegram bot that uses LangGraph and Ollama to provide adaptive RAG-based responses.

## Prerequisites

1. Python 3.9+
2. Ollama installed and running locally
3. Telegram Bot Token (get it from [@BotFather](https://t.me/botfather))
4. Conda installed

## Setup

1. Install Ollama and pull the Mistral model:
```bash
ollama pull mistral:7b-instruct
```

2. Create a conda environment and install dependencies:
```bash
conda create -n adaptive_rag python=3.10
conda activate adaptive_rag
pip install -r requirements.txt
```

3. Create a `.env` file with your Telegram bot token:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

4. Run the bot:
```bash
python main.py
```

## Usage

1. Start a chat with your bot on Telegram
2. Send any message to the bot
3. The bot will process your message using the LangGraph agent and respond with streaming updates 