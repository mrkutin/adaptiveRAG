import logging
from typing import Annotated, Sequence
from aiogram import Bot
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.prompts import ChatPromptTemplate


from config import settings
from retriever import OpenSearchRetriever

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
        self.retriever = OpenSearchRetriever()
        self.llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            timeout=settings.ollama_timeout,
            streaming=True,
            max_tokens=settings.ollama_max_tokens
            # temperature=0.1,    # Low temperature for consistent pattern matching
            # top_k=3,           # Very limited options for precise matching
            # top_p=0.1,         # High precision in pattern identification
            # num_ctx=8192,      # Large context to analyze many logs at once
            # repeat_penalty=1.2  # Higher penalty to avoid repetitive patterns
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
                Count the number of context documents.
                Question: {question} 
                Context: {context} 
                Answer:
            """
        )

        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents for the query."""
        try:
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text="Retrieving documents..."
            )

            # Get the last user message
            query = state["messages"][-1].content
            print(f"================ CHAIN QUERY: {query}")
            
            # Retrieve documents
            docs = self.retriever.invoke(query)
            await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
                            text=f"Retrieved {len(docs)} documents"
                        )
            # Update state with retrieved documents
            state["documents"] = docs
            
            return state
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            raise
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def process_message(self, state: ChatState) -> ChatState:
        """Process a message and update the state."""
        try:
            # Get the user message from history
            last_message = state["messages"][-1].content
            first_message = state["messages"][0].content

            # Send initial "processing" message
            msg = await self.bot.send_message(
                chat_id=state["telegram_chat_id"],
                text=f"Answering your question: {first_message}"
            )

            current_sentence = ""
            full_response = ""
            
            # Stream the response
            formatted_docs = self.format_docs(state["documents"])
            async for chunk in self.chain.astream({"context": formatted_docs, "question": first_message}):
                if chunk:
                    current_sentence += chunk
                    
                    # Check if we have a complete sentence
                    if any(current_sentence.strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                        full_response += current_sentence
                        current_sentence = ""
                        # Update the message with the accumulated response
                        await self.bot.edit_message_text(
                            chat_id=state["telegram_chat_id"],
                            message_id=msg.message_id,
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
        self.workflow.add_node("retrieve", self.chat_chain.retrieve_documents)
        self.workflow.add_node("answer", self.chat_chain.process_message)
        
        # Add edges
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "answer")
        self.workflow.add_edge("answer", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    async def process(self, initial_state: ChatState):
        """Process a message through the workflow."""
        return await self.app.ainvoke(initial_state) 
    


# print(ChatChain(bot=Bot(token=settings.telegram_bot_token)).prompt.format(question="What are Mindbox upload server errors in topic id-authorize-customer-topic?", context=))