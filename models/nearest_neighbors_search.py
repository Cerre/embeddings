import numpy as np
from sklearn.neighbors import NearestNeighbors

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
