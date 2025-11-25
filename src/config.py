"""
Configuration for Sodick RAG Assistant
Pinecone database already contains the data
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Verify keys are loaded
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

print(f"Pinecone API Key: {'✅ Loaded' if PINECONE_API_KEY else '❌ Missing'}")
print(f"OpenAI API Key: {'✅ Loaded' if OPENAI_API_KEY else '❌ Missing'}")

# Set environment variables (required by some libraries)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =============================================================================
# PINECONE CONFIGURATION
# =============================================================================
INDEX_NAME = "sodic-en"  # Your existing index name
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU available

# Model configuration
EMBEDDING_MODEL_KWARGS = {
    'device': EMBEDDING_DEVICE,
    'trust_remote_code': True
}

EMBEDDING_ENCODE_KWARGS = {
    'normalize_embeddings': True
}

# =============================================================================
# LLM CONFIGURATION (OpenAI)
# =============================================================================
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# =============================================================================
# RAG RETRIEVAL CONFIGURATION
# =============================================================================
RETRIEVAL_SEARCH_TYPE = "similarity"
RETRIEVAL_TOP_K = 5  # Number of documents to retrieve

# =============================================================================
# CHAT HISTORY CONFIGURATION
# =============================================================================
MAX_CHAT_HISTORY = 10  # Keep last 10 messages (5 Q&A pairs)

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
print("\n" + "="*70)
print("📋 SODICK RAG CONFIGURATION")
print("="*70)
print(f"Pinecone Index: {INDEX_NAME}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSION}D)")
print(f"LLM: OpenAI (temp={LLM_TEMPERATURE}, max_tokens={LLM_MAX_TOKENS})")
print(f"Retrieval: Top-{RETRIEVAL_TOP_K} ({RETRIEVAL_SEARCH_TYPE})")
print(f"Chat History: Last {MAX_CHAT_HISTORY} messages")
print("="*70 + "\n")
