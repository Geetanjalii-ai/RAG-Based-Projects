""" STEP 6: Connect to the LLM(the brain of the assisant)."""

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq

from hr_assistant import config

def get_llm():

    """ Return a groq chat model.Reads GROQ_API_KEY from the env"""

    return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)

