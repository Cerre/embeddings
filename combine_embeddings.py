import json
import os

# Define the path to the directory containing embedding files
embeddings_dir = "embeddings"

# Load the combined JSON file
with open('combined_transcriptions_2.json', 'r') as file:
    data = json.load(file)

# Function to load an embedding from a file
def load_embedding(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Iterate over each video in the combined JSON
for video in data:
    video_id = video['videoId']
    
    # Iterate over each text chunk in the video
    for idx, chunk in enumerate(video['text_chunks']):
        # Construct the filename for the corresponding embedding
        embedding_file = os.path.join(embeddings_dir, f"{video_id}_embedding_{idx}.json")
        
        # Load the embedding if the file exists
        if os.path.exists(embedding_file):
            chunk['embedding'] = load_embedding(embedding_file)
        else:
            print(f"Warning: Embedding file not found for {video_id}, chunk {idx}")

# Save the updated data to a new JSON file
with open('combined_transcriptions_with_embeddings.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Embeddings integrated and saved to combined_transcriptions_with_embeddings.json.")
