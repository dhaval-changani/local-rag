from config import Config
from embeddings import Embeddings
from retrieval import Retrieval
from llm import LLM
import click


@click.command()
@click.option("--input")
@click.option("--output")
def generate(input, output):
    config = Config(input, output)
    embeddings = Embeddings(config)
    embeddings.generate_embeddings()


@click.command()
@click.option("--sentence")
def retrieve(sentence: str):
    config = Config()
    retrieval = Retrieval(config)
    retrieval.retrieve(sentence, 5)


@click.command()
@click.option("--query")
def query(query: str):
    config = Config()
    llm = LLM(config, "You are a helpful assistant")
    llm.generate_response(query)
