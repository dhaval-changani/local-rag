from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pickle
import numpy as np
import os
from config import Config
from embeddings import Embeddings

load_dotenv()

model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL")


class Retrieval():

    def __init__(self, config: Config) -> None:
        self.config = config
        self.embeddings = Embeddings(config)
        self.model = self.embeddings.model
        self.store = self.embeddings.load_embeddings()

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        text = self.store['text']
        embeddings = self.store['embeddings']
        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"text": text[i], "score": scores[i]} for i in top_indices]
