"""
Helper functions for Sodick RAG Assistant
Contains retriever setup and chat functionality
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableMap
import re


# Import configuration
from src.config import (
    INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_KWARGS,
    EMBEDDING_ENCODE_KWARGS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RETRIEVAL_SEARCH_TYPE,
    RETRIEVAL_TOP_K,
    MAX_CHAT_HISTORY,
)

# Import prompt
from src.prompt import SYSTEM_PROMPT

# GLOBAL CHAT HISTORY
chat_history: List = []


# INITIALIZE EMBEDDINGS

def get_embeddings():
    """
    Initialize and return HuggingFace embeddings model
    Uses BAAI/bge-small-en-v1.5 (384 dimensions)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
        cache_folder=None,
    )

    print(f"Embeddings loaded: {EMBEDDING_MODEL_NAME}")
    return embeddings


# INITIALIZE RETRIEVER

def get_retriever():
    """
    Connect to existing Pinecone index and return retriever
    """
    embeddings = get_embeddings()

    # Connect to existing Pinecone index
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # Create retriever
    retriever = docsearch.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE, search_kwargs={"k": RETRIEVAL_TOP_K}
    )

    print(f"Connected to Pinecone index: {INDEX_NAME}")
    print(
        f"Retriever configured: Top-{RETRIEVAL_TOP_K} {RETRIEVAL_SEARCH_TYPE} search\n"
    )

    return retriever


# INITIALIZE LLM

def get_llm():
    """Initialize OpenAI LLM"""
    llm = ChatOpenAI(model=LLM_MODEL ,temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)

    print(
        f"LLM initialized: OpenAI (temp={LLM_TEMPERATURE}, max_tokens={LLM_MAX_TOKENS})"
    )
    return llm


# BUILD RAG CHAIN

def build_rag_chain():
    """
    Build the complete RAG chain with chat history support
    """
    retriever = get_retriever()
    llm = get_llm()

    # Create prompt template with chat history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # Retrieve context function
    def retrieve_context(inputs):
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    # Build the RAG chain
    rag_chain = (
        RunnableMap(
            {
                "context": retrieve_context,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
        )
        | prompt
        | llm
    )

    print("RAG chain built successfully\n")
    return rag_chain


# Initialize the RAG chain (singleton)
_rag_chain = None


def get_rag_chain():
    """Get or create RAG chain (singleton pattern)"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain()
    return _rag_chain


# CHAT FUNCTION WITH HISTORY

def chat(question: str) -> str:
    """
    Chat with the RAG system with conversation history

    Args:
        question: User's question

    Returns:
        str: Assistant's response
    """
    global chat_history

    rag_chain = get_rag_chain()

    # Invoke RAG chain with chat history (limit to last 10 messages)
    result = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history[-MAX_CHAT_HISTORY:],
        }
    )

    # Handle both string and message object responses
    if isinstance(result, str):
        response_content = result
    else:
        response_content = result.content if hasattr(result, "content") else str(result)

    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response_content))

    return response_content


# CHAT HISTORY MANAGEMENT

def clear_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    print("Chat history cleared")


def view_history():
    """View chat history (first 100 chars of each message)"""
    global chat_history

    if not chat_history:
        print("No chat history yet")
        return

    print(f"\nChat History ({len(chat_history)} messages):")
    print("=" * 70)

    for i, msg in enumerate(chat_history):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i + 1}. {role}: {content}")

    print("=" * 70 + "\n")


def get_history_count() -> int:
    """Get number of messages in history"""
    return len(chat_history)


def get_history() -> List:
    """Get the full chat history"""
    return chat_history


# GENERATE FOLLOW-UP QUESTIONS
def generate_followup_questions(last_question: str, last_answer: str) -> List[str]:
    """
    Generate contextual follow-up questions based on the conversation

    Args:
        last_question: User's last question
        last_answer: Assistant's last answer

    Returns:
        List of 5 follow-up questions
    """
    llm = get_llm()

    followup_prompt = f"""Based on this conversation about Sodick, generate 5 relevant follow-up questions that the user might want to ask next.

User's Question: {last_question}

Assistant's Answer: {last_answer[:500]}...

Generate 5 SHORT, specific follow-up questions (max 10 words each) that naturally continue this conversation. Focus on:
- Clarifying details mentioned in the answer
- Related products or features
- Technical specifications
- Practical applications
- Next steps or additional information

Format: Return ONLY the questions, one per line, numbered 1-5.

Example output:
1. What is the warranty period?
2. Tell me about the maintenance requirements
3. What industries use this product?
4. How does it compare to competitors?
5. Where can I purchase replacement parts?

Now generate 5 follow-up questions:"""

    try:
        result = llm.invoke(followup_prompt)
        
        #: Extract content from AIMessage object
        response_text = result.content if hasattr(result, 'content') else str(result)
        
        # Parse the response
        lines = response_text.strip().split("\n")
        questions = []

        for line in lines:
            # Remove numbering and clean up
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned and len(cleaned) > 10:  # Valid question
                questions.append(cleaned)

        # Return top 5, or use fallback if parsing failed
        if len(questions) >= 3:
            return questions[:5]
        else:
            print(f"Only got {len(questions)} questions, using fallback")
            return [
                "Tell me more about that",
                "What are the specifications?",
                "How much does it cost?",
                "Where can I learn more?",
                "What are the alternatives?",
            ]

    except Exception as e:
        print(f"Error generating follow-ups: {e}")
        # Return generic follow-ups on error
        return [
            "Can you provide more details?",
            "What are related products?",
            "How do I contact support?",
            "What industries is this for?",
            "Tell me about pricing",
        ]
