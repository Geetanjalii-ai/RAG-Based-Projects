"""All Settings for the app live here,in one place."""

import os
# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

load_dotenv()


#ENV VAR/ SECRET
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
JINA_API_KEY=os.getenv("JINA_API_KEY")

print("Enviroment Variables loaded!!")

#DEFINE PATH-DATA/VECTOR STORE

DATA_FILE_PATH=os.path.join("data","hr-policy.txt")

VECTOR_STORE_PATH=os.path.join("data","faiss_index")

##MODELS

LLM_MODEL_NAME="openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME="jina-embeddings-v2-base-en"

#CHUNK/TEXT SPLITTINH CONFIG
CHUNK_SIZE=500
CHUNK_OVERLAP=50

#RETRIVAL RESULTS
TOP_K_RESULTS=3

#SYSTEM INSTRUCTIONS

SYSTEM_PROMPT=(
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."

)

#Tracing
# TRACING 

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")


def check_api_keys()->None:

    """stop early with a clear message if a required API key is not present. """
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY,Please add it to your .env")
    if not JINA_API_KEY:
        raise ValueError("Missing GROQ_API_KEY,Please add it to your .env")









