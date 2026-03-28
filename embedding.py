import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


def parse_text_into_sentences(fileName: str):
    sentences = []
    file = open(fileName)
    lines = file.readlines()

    for line in lines:
        if line != "\n":
            line = line.replace("\n", "")
            sentences.append(line)

    return sentences


def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


sentences = parse_text_into_sentences("data.txt")

model = load_model()

embeddings = model.encode(sentences)

sim_1 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_2 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f'similar sentences: {sim_1:.3f}')
print(f'unrelated sentences: {sim_2:.3f}')
