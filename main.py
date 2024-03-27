import os
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from typing import Tuple
from models.embedding_generator import OpenAIEmbeddingGenerator
from models.nearest_neighbors_search import NearestNeighborsSearch
from services.llm_handler import LLMHandler
from openai import OpenAI

app = FastAPI()


class Query(BaseModel):
    text: str


# Initialize your classes outside the endpoint function to avoid re-initializing them on each request
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()  # Make sure to securely manage your API key
model = "text-embedding-3-large"
embedding_generator = OpenAIEmbeddingGenerator(openai_client, model)
nn_search = NearestNeighborsSearch()
data_path = "data/combined_transcriptions_with_embeddings_text-embedding-3-large.json"
nn_search.load_data(data_path)  # Assume this is preloaded with your embeddings data
llm_handler = LLMHandler(model)
youtube_url_watch = "https://www.youtube.com/watch?v"


@app.post("/find_best_match/")
async def find_best_match(query: Query):
    try:
        # Generate the query embedding
        query_embedding = embedding_generator.generate_embedding(query.text)

        # Find the nearest neighbors
        nearest_info = nn_search.find_nearest(np.array([query_embedding]))

        # Use the LLM to determine the best match
        video_id, timestamp, output_text = llm_handler.find_best_match(
            query.text, nearest_info
        )
        timestamp_formatted = format_timestamp(timestamp)
        # Assuming best_match returns video_id and timestamp
        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={timestamp_formatted}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# The process_query function is now integrated within the find_best_match endpoint function.


def format_timestamp(timestamp: str) -> str:
    """
    Dynamically format a timestamp, appending 'h', 'm', and 's' to the respective parts.
    Handles both HH:MM:SS and MM:SS formats correctly, omitting leading zeros for hours.

    Args:
    - timestamp (str): The timestamp in HH:MM:SS or MM:SS format.

    Returns:
    - str: A formatted duration string, e.g., "1h30m30s" or "23m45s".
    """
    parts = timestamp.split(":")
    formatted_parts = []

    # Depending on the number of parts, append the correct suffix.
    if len(parts) == 3:
        # If hours are present
        if int(parts[0]) > 0:  # Only append hours if it's more than 0
            formatted_parts.append(parts[0] + "h")
        formatted_parts.append(parts[1] + "m")
    elif len(parts) == 2:
        # If hours are not present
        formatted_parts.append(parts[0] + "m")
    else:
        raise ValueError("Timestamp format is incorrect. Expected HH:MM:SS or MM:SS.")

    # Seconds are always present and processed the same way
    formatted_parts.append(parts[-1] + "s")

    return "".join(formatted_parts)
