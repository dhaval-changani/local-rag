import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

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


def save_embeddings(sentences, embeddings, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump({"text": sentences, "embeddings": embeddings}, f)


sentences = parse_text_into_sentences("data.txt")

model = load_model()

embeddings = model.encode(sentences)

save_embeddings(sentences, embeddings, "pickle_dup.pkl")
