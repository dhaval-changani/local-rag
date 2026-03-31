from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pickle
from config import Config

load_dotenv()


class Embeddings:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: SentenceTransformer = SentenceTransformer(
            self.config.modelName)

    def parse_text_into_sentences(self, fileName: str) -> list[str]:
        sentences = []
        with open(fileName) as f:
            lines = f.readlines()

        for line in lines:
            if line != "\n":
                line = line.replace("\n", "")
                sentences.append(line)

        return sentences

    def save_embeddings(self, sentences, embeddings) -> None:
        with open(self.config.embeddingFile, 'wb') as f:
            pickle.dump({"text": sentences, "embeddings": embeddings}, f)

    def load_embeddings(self):
        with open(self.config.embeddingFile, 'rb') as f:
            return pickle.load(f)

    def generate_embeddings(self) -> None:
        sentences = self.parse_text_into_sentences(self.config.inputFile)
        embeddings = self.model.encode(sentences)
        self.save_embeddings(sentences, embeddings)
