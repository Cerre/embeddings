import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class NearestNeighborsSearch:
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean'):
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        self.embeddings = None
        self.video_ids = None
        self.text_chunks_info = None

    def fit(self, embeddings: np.ndarray):
        self.nn.fit(embeddings)

    def find_nearest(self, query_embedding: np.ndarray):
        distances, indices = self.nn.kneighbors(query_embedding)
        nearest_info = [(self.video_ids[idx], self.text_chunks_info[idx][0], self.text_chunks_info[idx][1]) for idx in indices[0]]
        return nearest_info


    def load_data(self, data_path):
        df = pd.read_json(data_path)
        embeddings = []
        video_ids = []
        text_chunks_info = []

        for _, row in df.iterrows():
            for chunk in row['text_chunks']:
                embeddings.append(chunk['embedding'])
                video_ids.append(row['videoId'])
                text_chunks_info.append((chunk['start_time'], chunk['chunk_text']))


        embeddings_np = np.array(embeddings)
        self.video_ids = video_ids
        self.text_chunks_info = text_chunks_info
        # Assume embeddings_np is available
        self.fit(embeddings_np)