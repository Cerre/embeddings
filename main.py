from models.embedding_generator import OpenAIEmbeddingGenerator
from models.nearest_neighbors_search import NearestNeighborsSearch
from services.llm_handler import LLMHandler
from utils.metadata_manager import MetadataManager

import numpy as np
import pandas as pd
import os
from openai import OpenAI
from pathlib import Path


def main():
    # Assuming client and generator setup
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
    model = "text-embedding-3-large"
    embedding_generator = OpenAIEmbeddingGenerator(client, model)

    metadata_file = Path('metadata.json')
    metadata_manager = MetadataManager(metadata_file)
    nn_search = NearestNeighborsSearch()
    llm_handler = LLMHandler(model)
    embeddings_file = Path('vector_database.ann')

    if embeddings_file.exists() and metadata_file.exists():
        # Load the metadata
        video_ids, text_chunks_info = metadata_manager.load_metadata()
        # Load your embeddings directly, since we're assuming they're precomputed
        # This part is left as an exercise to the reader, as it depends on how you've stored your embeddings
    else:
        # df = pd.read_json('data/combined_transcriptions_with_embeddings.json')
        df = pd.read_json('data/combined_transcriptions_with_embeddings_text-embedding-3-large.json')
        
        embeddings = []
        video_ids = []
        text_chunks_info = []

        for _, row in df.iterrows():
            for chunk in row['text_chunks']:
                embeddings.append(chunk['embedding'])
                video_ids.append(row['videoId'])
                text_chunks_info.append((chunk['start_time'], chunk['chunk_text']))

        # Save the metadata for future use
        metadata_manager.save_metadata(video_ids, text_chunks_info)

    embeddings_np = np.array(embeddings)
    nn_search.video_ids = video_ids
    nn_search.text_chunks_info = text_chunks_info
    # Assume embeddings_np is available
    nn_search.fit(embeddings_np)

    # Example query
    input_text = "it's the worst one v one keeping the ball up history"
    input_text = "worst we've seen in keeping the ball up history in one v one"
    input_text = "worst one v one keeping the ball up"
    input_text = "keeping the ball up competition fail"
    input_text = "keep ball up fail"
    # input_text = "zen is scoring against mawkzy"
    # input_text = "someone says disgusting several times"
    query_embedding = embedding_generator.generate_embedding(input_text)
    nearest_info = nn_search.find_nearest(np.array([query_embedding]))

    for video_id, timestamp, text in nearest_info:
        print(f"Video ID: {video_id}, Timestamp: {timestamp}, Text: {text}")

    best_match = llm_handler.find_best_match(input_text, nearest_info)

    video_id, timestamp, output_text = best_match
    print(f"Best Match Video ID: {video_id}, Timestamp: {timestamp}, Text: {output_text}")


if __name__ == "__main__":
    main()
