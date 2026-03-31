import os
from dotenv import load_dotenv

load_dotenv()


class Config():

    def __init__(self) -> None:
        self.modelName = os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.local_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.inputFile = os.getenv("INPUT_FILE", 'data.txt')
        self.embeddingFile = os.getenv("EMBEDDING_FILE", 'embedding.pkl')
