from langchain_openai import ChatOpenAI
from load_env import openai_api_key
from load_env import openai_base_url
def get_llm():
    """Loads LLM Securely."""
    return ChatOpenAI(
       model="openai/gpt-oss-120b:free",
       api_key=openai_api_key,
       base_url=openai_base_url,
       temperature=0,
       max_tokens=500

    )
