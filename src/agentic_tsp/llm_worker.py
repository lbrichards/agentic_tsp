# In src/agentic_tsp/llm_worker.py
import os
import json
from dotenv import load_dotenv
import openai
from typing import List, Tuple

# Load environment variables from .env
load_dotenv()

def get_openai_client():
    """Initializes and returns the OpenAI client, ensuring API key is set."""
    api_key = os.getenv("OA_API_KEY")
    if not api_key:
        raise ValueError("OA_API_KEY not found in .env file. Please run setup_env.py")
    return openai.OpenAI(api_key=api_key)

def build_llm_prompt(tour: List[int], cities: List[Tuple[float, float]]) -> str:
    """Builds the detailed prompt for the LLM."""
    
    prompt = f"""
    You are an expert in combinatorial optimization, specifically the Traveling Salesperson Problem (TSP).
    Your task is to improve a given tour of cities. I will provide you with the city coordinates and the current tour sequence.

    A common method for improving a TSP tour is the 2-opt heuristic, which involves finding two edges that cross in the tour path and swapping them to uncross them. This almost always reduces the total tour length.

    Here are the cities, identified by an ID (their index) and their (x, y) coordinates:
    { {i: coord for i, coord in enumerate(cities)} }

    Here is the current tour, represented by the sequence of city IDs:
    {tour}

    Analyze this tour. Identify any obvious crossings or poorly ordered segments.
    Your goal is to propose a new tour that is shorter.
    Think step-by-step about which pairs of edges could be swapped to improve the path.

    Return ONLY a JSON object with a single key "improved_tour" containing the new sequence of city IDs.
    Do not include any other text, explanation, or markdown.

    Example response format:
    {{
      "improved_tour": [0, 5, 2, 8, ...]
    }}
    """
    return prompt

def llm_improve_tour(tour: List[int], cities: List[Tuple[float, float]]) -> List[int]:
    """
    Uses an LLM to attempt to improve a given TSP tour.
    Returns a new tour list, or the original tour if the process fails.
    """
    client = get_openai_client()
    prompt = build_llm_prompt(tour, cities)
    
    print("\n--- Sending tour to LLM for improvement... ---")
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        print("--- LLM Response Received ---")
        
        # Parse the JSON response to extract the improved tour
        json_response = json.loads(response_content)
        improved_tour = json_response.get("improved_tour")
        
        # Basic validation of the returned tour
        if improved_tour and isinstance(improved_tour, list) and set(improved_tour) == set(tour):
            print("LLM returned a valid improved tour.")
            return improved_tour
        else:
            print("Warning: LLM returned an invalid or malformed tour. Using original tour.")
            return tour

    except Exception as e:
        print(f"An error occurred during LLM API call: {e}")
        return tour