from sklearn.neighbors import NearestNeighbors
import numpy as np
from openai import OpenAI
import pandas as pd
import os
import json
from pathlib import Path
from models.embedding_generator import OpenAIEmbeddingGenerator

# Assuming the existence of generate_embedding function from create_embeddings module
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
embedding_generator = OpenAIEmbeddingGenerator(client)
# Check if the embeddings and metadata files exist
embeddings_file = Path("vector_database.ann")
metadata_file = Path("metadata.json")


# Function to save metadata
def save_metadata(video_ids, text_chunks_info):
    with open(metadata_file, "w") as f:
        json.dump({"video_ids": video_ids, "text_chunks_info": text_chunks_info}, f)


# Function to load metadata
def load_metadata():
    with open(metadata_file, "r") as f:
        data = json.load(f)
    return data["video_ids"], data["text_chunks_info"]


# Assuming the rest of your setup code remains the same

if embeddings_file.exists() and metadata_file.exists():
    # Load the metadata
    video_ids, text_chunks_info = load_metadata()
    # Load your embeddings directly, since we're assuming they're precomputed
    # This part is left as an exercise to the reader, as it depends on how you've stored your embeddings
else:
    df = pd.read_json("data/combined_transcriptions_with_embeddings.json")
    embeddings = []
    video_ids = []
    text_chunks_info = []

    for _, row in df.iterrows():
        for chunk in row["text_chunks"]:
            embeddings.append(chunk["embedding"])
            video_ids.append(row["videoId"])
            text_chunks_info.append((chunk["start_time"], chunk["chunk_text"]))

    # Save the metadata for future use
    save_metadata(video_ids, text_chunks_info)

# Convert embeddings list to a numpy array for scikit-learn
embeddings_np = np.array(embeddings)
n_neighbors = 5
# Initialize NearestNeighbors with 'auto' algorithm, which will attempt to decide the most appropriate algorithm based on the values passed to fit method
nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")
nn.fit(embeddings_np)

# Generate or load your query embedding
text = "I swear could have done better in one v one  history, an absolute upset, both players should be embarrassed. complete failure"
text = "worst we've seen in keeping the ball up history"
query_embedding = embedding_generator.generate_embedding(text)
query_embedding_np = np.array(
    [query_embedding]
)  # Needs to be reshaped for a single query

# Find the exact nearest neighbors
distances, indices = nn.kneighbors(query_embedding_np)

# Retrieve information for the nearest neighbors
nearest_info = [
    (video_ids[idx], text_chunks_info[idx][0], text_chunks_info[idx][1])
    for idx in indices[0]
]

for video_id, timestamp, text in nearest_info:
    print(f"Video ID: {video_id}, Timestamp: {timestamp}, Text: {text}\n")
