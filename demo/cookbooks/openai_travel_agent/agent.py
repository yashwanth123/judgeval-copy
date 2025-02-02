import openai
import requests
import os
from tavily import TavilyClient
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import json

def populate_vector_db(collection, destinations_data):
    """
    Populate the vector DB with travel information.
    destinations_data should be a list of dictionaries with 'destination' and 'information' keys
    """
    for data in destinations_data:
        collection.add(
            documents=[data['information']],
            metadatas=[{"destination": data['destination']}],
            ids=[f"destination_{data['destination'].lower().replace(' ', '_')}"]
        )

def search_tavily(query):
    """Fetch travel data using Tavily API."""
    API_KEY = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=API_KEY)
    results = client.search(query, num_results=3)
    return results

def get_attractions(destination):
    """Search for top attractions in the destination."""
    return search_tavily(f"Best tourist attractions in {destination}")

def get_hotels(destination):
    """Search for hotels in the destination."""
    return search_tavily(f"Best hotels in {destination}")

def get_flights(destination):
    """Search for flights to the destination."""
    return search_tavily(f"Flights to {destination} from major cities")

def get_weather(destination, start_date, end_date):
    """Search for weather information."""
    return search_tavily(f"Weather forecast for {destination} from {start_date} to {end_date}")

def initialize_vector_db():
    """Initialize ChromaDB with OpenAI embeddings."""
    client = chromadb.Client()
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    return client.get_or_create_collection(
        "travel_information",
        embedding_function=embedding_fn
    )

def query_vector_db(collection, destination, k=3):
    """Query the vector database for existing travel information."""
    try:
        results = collection.query(
            query_texts=[destination],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []
    except Exception:
        return []

def research_destination(destination, start_date, end_date):
    """Gather all necessary travel information for a destination."""
    # First, check the vector database
    collection = initialize_vector_db()
    existing_info = query_vector_db(collection, destination)
    
    # Get real-time information from Tavily
    tavily_data = {
        "attractions": get_attractions(destination),
        "hotels": get_hotels(destination),
        "flights": get_flights(destination),
        "weather": get_weather(destination, start_date, end_date)
    }
    
    return {
        "vector_db_results": existing_info,
        **tavily_data
    }

def create_travel_plan(destination, start_date, end_date, research_data):
    """Generate a travel itinerary using the researched data."""
    vector_db_context = "\n".join(research_data['vector_db_results']) if research_data['vector_db_results'] else "No pre-stored information available."
    
    prompt = f"""
    Create a structured travel itinerary for a trip to {destination} from {start_date} to {end_date}.
    
    Pre-stored destination information:
    {vector_db_context}
    
    Current travel data:
    - Attractions: {research_data['attractions']}
    - Hotels: {research_data['hotels']}
    - Flights: {research_data['flights']}
    - Weather: {research_data['weather']}
    """
    
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert travel planner. Combine both historical and current information to create the best possible itinerary."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def generate_itinerary(destination, start_date, end_date):
    """Main function to generate a travel itinerary."""
    research_data = research_destination(destination, start_date, end_date)
    return create_travel_plan(destination, start_date, end_date, research_data)

if __name__ == "__main__":
    load_dotenv()
    destination = input("Enter your travel destination: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    itinerary = generate_itinerary(destination, start_date, end_date)
    print("\nGenerated Itinerary:\n", itinerary)
