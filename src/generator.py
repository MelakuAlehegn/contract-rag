from langchain.chat_models import ChatOpenAI
from src.config import OPENAI_API_KEY

def create_generator():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")