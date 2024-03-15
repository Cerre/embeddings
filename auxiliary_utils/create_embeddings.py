import openai
from dotenv import load_dotenv
import os
import json

# Load the API key from an .env file


from openai import OpenAI




def generate_embedding(client, text, model):
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    embedding = response.data[0].embedding  # Accessing the embedding data
    return embedding

def save_embedding_to_file(embedding, file_path):
    with open(file_path, 'w') as f:
        json.dump(embedding, f)

def generate_and_save_embedding(client, text, video_id, idx, model):
    try:
        embedding = generate_embedding(client, text, model)

        # Define the file path for saving the embedding
        file_path = f'embeddings_{model}/{video_id}_embedding_{idx}.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the embedding to a JSON file
        save_embedding_to_file(embedding, file_path)

        print(f'Embedding saved to {file_path}')
    except Exception as e:
        print(f'Error generating embedding: {e}')


# Load data from a JSON file
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data




def main():
    load_dotenv()
    # Example usage: Replace 'path_to_your_file.json' with the actual path to your JSON file
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(os.getenv("OPENAI_API_KEY"))
    client = OpenAI()
    file_path = 'data/combined_transcriptions_2.json'
    data = load_data_from_json(file_path)
    model = "text-embedding-ada-002"
    model = "text-embedding-3-large"

    # Check if 'text_chunks' exists in the data
    for i, item in enumerate(data):
        if 'text_chunks' in item:
            # Iterate through the text chunks and generate/save embeddings
            for idx, chunk in enumerate(item['text_chunks']):
                print(i, idx)
                generate_and_save_embedding(client, chunk['chunk_text'], item['videoId'], idx, model)
        else:
            print(f"No text_chunks found in the item for videoId: {item.get('videoId')}")

if __name__ == '__main__':
    main()
