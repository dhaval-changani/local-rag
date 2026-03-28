from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pickle
import numpy as np
import os

load_dotenv()

model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL")


def load_embeddings(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


store = load_embeddings("pickle_dup.pkl")
text = store["text"]
embeddings = store["embeddings"]

model = SentenceTransformer(model_name)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": text[i], "score": float(scores[i])} for i in top_indices
    ]


results = retrieve("what are store types?", 5)

for r in results:
    print(f"Text: {r['text'][:100]}, Score: {r['score']:.4f}\n")
