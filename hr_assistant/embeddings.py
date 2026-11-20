""" step 3: Turn text into numbers(vectors) using jina."""

# pyrefly: ignore[missing-import]
from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings():

    """
     Returns jina embeddings using JINA API KEY
    """

    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)