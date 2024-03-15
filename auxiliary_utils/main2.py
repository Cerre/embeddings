import pandas as pd
from annoy import AnnoyIndex
import os
import json
from pathlib import Path

# Assuming the existence of generate_embedding function from create_embeddings module
from create_embeddings import generate_embedding

# Check if the embeddings and metadata files exist
embeddings_file = Path('vector_database.ann')
metadata_file = Path('metadata.json')

# Function to save metadata
def save_metadata(video_ids, text_chunks_info):
    with open(metadata_file, 'w') as f:
        json.dump({'video_ids': video_ids, 'text_chunks_info': text_chunks_info}, f)

# Function to load metadata
def load_metadata():
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    return data['video_ids'], data['text_chunks_info']

if embeddings_file.exists() and metadata_file.exists():
    # Load the metadata
    video_ids, text_chunks_info = load_metadata()
    f = len(text_chunks_info[0][1])  # Assuming the embedding size can be inferred from the first saved chunk
else:
    df = pd.read_json('combined_transcriptions_with_embeddings.json')
    embeddings = []
    video_ids = []
    text_chunks_info = []

    for _, row in df.iterrows():
        for chunk in row['text_chunks']:
            embeddings.append(chunk['embedding'])
            video_ids.append(row['videoId'])
            text_chunks_info.append((chunk['start_time'], chunk['chunk_text']))

    # Assuming all embeddings have the same dimensionality
    f = len(embeddings[0])
    t = AnnoyIndex(f, 'angular')
    
    for i, vec in enumerate(embeddings):
        t.add_item(i, vec)

    t.build(10)  # Adjust the number of trees as needed
    t.save('vector_database.ann')

    # Save the metadata for future use
    save_metadata(video_ids, text_chunks_info)

# Load the Annoy index
u = AnnoyIndex(f, 'angular')
u.load('vector_database.ann')  # Load your Annoy index

# Example query
text = "Your query text here"
# Load the indexed file
query_embedding = generate_embedding(client, text)  # Generate or load your query embedding as needed

# Finding the nearest neighbors
n_neighbors = 3
nearest_ids = u.get_nns_by_vector(query_embedding, n_neighbors)

# Retrieve information for the nearest neighbors
nearest_info = [(video_ids[i], text_chunks_info[i][0], text_chunks_info[i][1]) for i in nearest_ids]

for video_id, timestamp, text in nearest_info:
    print(f"Video ID: {video_id}, Timestamp: {timestamp}, Text: {text}\n")
