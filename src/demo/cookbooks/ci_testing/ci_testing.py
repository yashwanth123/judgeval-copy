"""
Cookbook for CI testing LLM applications using `judgeval`

Includes unit tests and end-to-end tests for an OpenAI API-based travel agent
"""

import asyncio
import os
import pytest
from demo.cookbooks.openai_travel_agent.agent import *
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    AnswerCorrectnessScorer, 
    AnswerRelevancyScorer, 
    FaithfulnessScorer
)   


@pytest.fixture
def judgment_client():
    return JudgmentClient()


@pytest.fixture
def research_data():
    return {
        "attractions": [
            "The iconic Eiffel Tower stands at 324 meters tall and welcomes over 7 million visitors annually. Visitors can access three levels, with the top floor offering panoramic views of Paris. The tower features two restaurants: 58 Tour Eiffel and the Michelin-starred Le Jules Verne.",
            "The Louvre Museum houses over 380,000 objects and displays 35,000 works of art across eight departments. Home to the Mona Lisa and Venus de Milo, it's the world's largest art museum with 72,735 square meters of exhibition space. Visitors typically need 3-4 hours to see the highlights.",
            "The historic district of Montmartre sits on Paris's highest hill at 130 meters. Famous for the white Sacré-Cœur Basilica and Place du Tertre filled with artists, it was once home to renowned painters like Picasso and Van Gogh. The area retains its village-like charm with winding cobblestone streets and authentic Parisian cafes."
        ],
        "hotels": [
            "Hotel de la Paix is a luxurious 5-star establishment in the 16th arrondissement, featuring 85 rooms and suites decorated in classic Parisian style. The hotel offers a Michelin-starred restaurant, spa facilities, and is located just 10 minutes from the Arc de Triomphe.",
            "Hotel de Paris, situated in the Opera district, combines Belle Époque architecture with modern amenities. Recently renovated in 2022, it offers 107 rooms, a rooftop bar with Eiffel Tower views, and has received the Palace distinction for exceptional service.",
            "Hotel de Ville, a boutique hotel in Le Marais, occupies a restored 17th-century mansion. With 40 individually designed rooms, a courtyard garden, and acclaimed restaurant, it provides an authentic Parisian experience steps from Notre-Dame Cathedral."
        ],
        "flights": [
            "Multiple daily direct flights to Paris Charles de Gaulle (CDG) from major US cities. Air France and United Airlines operate regular routes from JFK, LAX, and Chicago O'Hare. Flight times range from 7-11 hours depending on departure city.",
            "From San Francisco International Airport (SFO), Air France operates a daily direct flight AF085 departing at 3:30 PM, arriving at CDG at 11:15 AM next day. United Airlines also offers UA990 with a similar schedule. Average flight time is 10 hours 45 minutes."
        ],
        "weather": "Paris in mid-February typically experiences cool winter conditions with average daytime temperatures ranging from 8-12°C (46-54°F). Current forecast shows mostly sunny conditions with occasional cloud cover. Morning temperatures around 6°C (43°F) rising to 12°C (54°F) by afternoon. Light breeze of 8-12 km/h expected with 20% chance of precipitation. Evening temperatures dropping to 4°C (39°F). UV index moderate at 3.",
        "vector_db_results": []
    }


@pytest.fixture
def sample_itinerary() -> str:
    """
    Loads the sample itinerary from the saved file
    """
    PATH_TO_ITINERARY = os.path.join(os.path.dirname(__file__), "travel_response.txt")
    with open(PATH_TO_ITINERARY, 'r') as file:
        return file.read()


@pytest.fixture
def expected_itinerary():
    return """5-Day Paris Itinerary (February 11-15, 2025)

Accommodation: Hotel de Paris in the Opera district
- Selected for its central location, rooftop bar with Eiffel Tower views, and recent 2022 renovation

Transportation: 
- Arrival via Air France flight AF085/United Airlines UA990 from SFO, landing at CDG at 11:15 AM

Weather Considerations:
- Pack warm clothing for temperatures between 4-12°C (39-54°F)
- Morning activities planned indoors due to cooler temperatures
- Outdoor activities scheduled during peak afternoon warmth

Day 1 (Feb 11):
- 11:15 AM: Arrival at CDG, transfer to Hotel de Paris
- 2:00 PM: Hotel check-in and refresh
- 3:30 PM: Visit the Eiffel Tower (taking advantage of afternoon warmth)
- 7:00 PM: Dinner at Le Jules Verne in the Eiffel Tower

Day 2 (Feb 12):
- 9:00 AM: Breakfast at hotel
- 10:00 AM: Louvre Museum visit (3-4 hours, indoor activity during cool morning)
- 2:30 PM: Late lunch in Opera district
- 4:00 PM: Rooftop bar at hotel for sunset views
- Evening: Dinner at hotel's restaurant

Day 3 (Feb 13):
- 10:00 AM: Visit Montmartre (during warming temperatures)
- 11:00 AM: Explore Sacré-Cœur Basilica
- 12:30 PM: Lunch at local cafe in Montmartre
- 2:00 PM: Artist square at Place du Tertre
- Evening: Dinner at authentic Parisian bistro

Day 4 (Feb 14):
- Morning: Arc de Triomphe visit (10-minute walk from hotel)
- Afternoon: Shopping and exploring Opera district
- Evening: Valentine's Day dinner at hotel's Michelin-starred restaurant

Day 5 (Feb 15):
- Morning: Leisurely breakfast
- Late morning: Check-out and departure

Note: Indoor alternatives planned in case of precipitation (20% chance). Schedule optimized around temperature peaks of 12°C in afternoons."""


def test_websearch_tool_answer_relevancy(judgment_client):
    query = "What is the weather like in San Francisco on February 11th, 2025?"
    results = search_tavily(query)

    example = Example(
        input=query,
        actual_output=str(results)
    )

    scorer = AnswerRelevancyScorer(threshold=0.8)
    
    judgment_client.assert_test(
        examples=[example],
        scorers=[scorer],
        model="gpt-4o-mini",
        project_name="travel_agent_tests",
        eval_run_name="websearch_relevancy_test",
        override=True
    )


def test_travel_planning_faithfulness(judgment_client, sample_itinerary, research_data):
    
    destination = "Paris, France"
    start_date = "February 11th, 2025"
    end_date = "February 15th, 2025"

    hotels_example = Example(
        input=f"Create a structured travel itinerary for a trip to {destination} from {start_date} to {end_date}.",
        actual_output=sample_itinerary,
        retrieval_context=research_data["hotels"]
    )

    flights_example = Example(
        input=f"Create a structured travel itinerary for a trip to {destination} from {start_date} to {end_date}.",
        actual_output=sample_itinerary,
        retrieval_context=research_data["flights"]
    )

    judgment_client.assert_test(
        examples=[hotels_example, flights_example],
        scorers=[FaithfulnessScorer(threshold=1.0)],
        model="gpt-4o",
        project_name="travel_agent_tests",
        eval_run_name="travel_planning_faithfulness_test",
        override=True
    )


def test_travel_planning_answer_correctness(judgment_client, sample_itinerary, expected_itinerary):
    
    destination = "Paris, France"
    start_date = "February 11th, 2025"
    end_date = "February 15th, 2025"

    example = Example(
        input=f"Create a structured travel itinerary for a trip to {destination} from {start_date} to {end_date}.",
        actual_output=sample_itinerary,
        expected_output=expected_itinerary
    )
    with pytest.raises(AssertionError):
        judgment_client.assert_test(
            examples=[example],
            scorers=[AnswerCorrectnessScorer(threshold=0.75)],
            model="gpt-4o",
            project_name="travel_agent_tests",
            eval_run_name="travel_planning_correctness_test",
            override=True
        )


def save_travel_response(destination, start_date, end_date, research_data, file_path):
    response = asyncio.run(create_travel_plan(destination, start_date, end_date, research_data))
    with open(file_path, 'w') as f:
        f.write(response)


if __name__ == "__main__":
    sample_research_data = {
        "attractions": [
            "The iconic Eiffel Tower stands at 324 meters tall and welcomes over 7 million visitors annually. Visitors can access three levels, with the top floor offering panoramic views of Paris. The tower features two restaurants: 58 Tour Eiffel and the Michelin-starred Le Jules Verne.",
            "The Louvre Museum houses over 380,000 objects and displays 35,000 works of art across eight departments. Home to the Mona Lisa and Venus de Milo, it's the world's largest art museum with 72,735 square meters of exhibition space. Visitors typically need 3-4 hours to see the highlights.",
            "The historic district of Montmartre sits on Paris's highest hill at 130 meters. Famous for the white Sacré-Cœur Basilica and Place du Tertre filled with artists, it was once home to renowned painters like Picasso and Van Gogh. The area retains its village-like charm with winding cobblestone streets and authentic Parisian cafes."
        ],
        "hotels": [
            "Hotel de la Paix is a luxurious 5-star establishment in the 16th arrondissement, featuring 85 rooms and suites decorated in classic Parisian style. The hotel offers a Michelin-starred restaurant, spa facilities, and is located just 10 minutes from the Arc de Triomphe.",
            "Hotel de Paris, situated in the Opera district, combines Belle Époque architecture with modern amenities. Recently renovated in 2022, it offers 107 rooms, a rooftop bar with Eiffel Tower views, and has received the Palace distinction for exceptional service.",
            "Hotel de Ville, a boutique hotel in Le Marais, occupies a restored 17th-century mansion. With 40 individually designed rooms, a courtyard garden, and acclaimed restaurant, it provides an authentic Parisian experience steps from Notre-Dame Cathedral."
        ],
        "flights": [
            "Multiple daily direct flights to Paris Charles de Gaulle (CDG) from major US cities. Air France and United Airlines operate regular routes from JFK, LAX, and Chicago O'Hare. Flight times range from 7-11 hours depending on departure city.",
            "From San Francisco International Airport (SFO), Air France operates a daily direct flight AF085 departing at 3:30 PM, arriving at CDG at 11:15 AM next day. United Airlines also offers UA990 with a similar schedule. Average flight time is 10 hours 45 minutes."
        ],
        "weather": "Paris in mid-February typically experiences cool winter conditions with average daytime temperatures ranging from 8-12°C (46-54°F). Current forecast shows mostly sunny conditions with occasional cloud cover. Morning temperatures around 6°C (43°F) rising to 12°C (54°F) by afternoon. Light breeze of 8-12 km/h expected with 20% chance of precipitation. Evening temperatures dropping to 4°C (39°F). UV index moderate at 3.",
        "vector_db_results": []
    }

    save_travel_response("Paris, France", "February 11th, 2025", "February 15th, 2025", sample_research_data, "./travel_response.txt")
