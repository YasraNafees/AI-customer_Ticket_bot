import os
from config import logger
from dotenv import load_dotenv

 # loading api key 
load_dotenv()
openai_api_key=os.getenv("OPENROUTER_API_KEY_1")
openai_base_url=os.getenv("OPENROUTER_API_BASE")

if not openai_api_key:
    raise RuntimeError("openai api is not found .env")

VECTORSTORE_DIR="vectorstore"
logger.info(f"Vectorstore directory set to :{VECTORSTORE_DIR}")
