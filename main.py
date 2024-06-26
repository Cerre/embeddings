import os
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Tuple
from models.embedding_generator import OpenAIEmbeddingGenerator
from models.nearest_neighbors_search import NearestNeighborsSearch
from services.llm_handler import LLMHandler
from openai import OpenAI

app = FastAPI(
    title="Your API",
    description="API for finding best matches",
    version="1.0.0",
    openapi_tags=[{"name": "matches", "description": "Operations related to finding matches"}],
)

API_KEY = os.environ.get("API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY or not OPENAI_API_KEY:
    raise ValueError("API_KEY and OPENAI_API_KEY must be set in the environment variables")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate API key")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Your API",
        version="1.0.0",
        description="API for finding best matches",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

class Query(BaseModel):
    text: str

def initialize_components():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    model = "text-embedding-3-large"
    embedding_generator = OpenAIEmbeddingGenerator(openai_client, model)
    nn_search = NearestNeighborsSearch()
    data_path = "data/combined_transcriptions_with_embeddings_text-embedding-3-large.json"
    nn_search.load_data(data_path)
    llm_handler = LLMHandler(model)
    return embedding_generator, nn_search, llm_handler

embedding_generator, nn_search, llm_handler = initialize_components()
youtube_url_watch = "https://www.youtube.com/watch?v"

@app.post("/find_best_match/", tags=["matches"])
async def find_best_match(query: Query, api_key: str = Security(get_api_key)):
    try:
        query_embedding = embedding_generator.generate_embedding(query.text)
        nearest_info = nn_search.find_nearest(np.array([query_embedding]))
        video_id, timestamp, output_text = llm_handler.find_best_match(query.text, nearest_info)
        timestamp_formatted = format_timestamp(timestamp)
        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "url_with_timestamp": f"{youtube_url_watch}={video_id}&t={timestamp_formatted}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def format_timestamp(timestamp: str) -> str:
    parts = timestamp.split(":")
    formatted_parts = []
    if len(parts) == 3:
        if int(parts[0]) > 0:
            formatted_parts.append(parts[0] + "h")
        formatted_parts.append(parts[1] + "m")
    elif len(parts) == 2:
        formatted_parts.append(parts[0] + "m")
    else:
        raise ValueError("Timestamp format is incorrect. Expected HH:MM:SS or MM:SS.")
    formatted_parts.append(parts[-1] + "s")
    return "".join(formatted_parts)