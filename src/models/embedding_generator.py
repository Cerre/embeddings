from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI


class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_embedding(self, text):
        response = self.client.embeddings.create(model=self.model, input=[text])
        embedding = response.data[0].embedding  # Accessing the embedding data
        return embedding
