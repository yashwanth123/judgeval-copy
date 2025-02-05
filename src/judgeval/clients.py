import os
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from together import Together, AsyncTogether

PATH_TO_DOTENV = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=PATH_TO_DOTENV)

# Initialize clients
client = OpenAI()
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)
together_client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))
async_together_client = AsyncTogether(api_key=os.getenv("TOGETHERAI_API_KEY"))

