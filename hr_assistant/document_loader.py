"""Step 1: Read the raw document from the data folder"""

# pyrefly: ignore[missing-import]
from langchain_community.document_loaders import TextLoader
from hr_assistant.logger import get_logger
from hr_assistant import config

logger=get_logger(__name__)

def load_document(file_path:str=config.DATA_FILE_PATH):
    """Load a .txt file and return as a list of langchain document"""

    logger.info("LOADING DOCUMNETS FROM DOCUMENT LOADER",file_path)

    loader=TextLoader(file_path,encoding="utf-8")

    documents=loader.load() 

    logger.info("Loaded %d document(s)", len(documents))

    return documents


