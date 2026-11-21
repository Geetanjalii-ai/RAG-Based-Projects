""" step 3: Turn text into numbers(vectors) using jina."""

# pyrefly: ignore[missing-import]
from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def get_embeddings():

    """
     Returns jina embeddings using JINA API KEY
    """
    logger.info("Initializing embeddings model '%s'", config.EMBEDDING_MODEL_NAME)
    
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)