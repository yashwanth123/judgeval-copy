import openai
import os
import asyncio
from tavily import TavilyClient

def search_tavily(query):
    """Fetch travel data using Tavily API."""
    API_KEY = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=API_KEY)
    results = client.search(query, num_results=3)
    return results
