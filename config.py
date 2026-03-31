import os
from dotenv import load_dotenv

load_dotenv()


class Config():

    def __init__(self, input=None, output=None) -> None:
        self.modelName = os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.local_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.inputFile = input if input is not None else os.getenv(
            "INPUT_FILE", 'data.txt')
        self.embeddingFile = output if output is not None else os.getenv(
            "EMBEDDING_FILE", 'embedding.pkl')
