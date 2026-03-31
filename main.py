from config import Config
from embeddings import Embeddings
from retrieval import Retrieval
from llm import LLM

COMMANDS = {
    "generate": "Generate embeddings from the data file",
    "query":    "Start an interactive Q&A session (type 'clear' to reset history, 'exit' to leave)",
    "help":     "Show this help message",
    "exit":     "Exit the program",
}


def print_help():
    print("\nAvailable commands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")
    print()


def run_generate(config):
    print("Generating embeddings...")
    embeddings = Embeddings(config)
    embeddings.generate_embeddings()
    print("Done.")


def run_query(retrieval, llm):
    print("Query mode started. Type 'clear' to reset history, 'exit' to return to main menu.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            llm.clear()
            print("Conversation history cleared.")
            continue

        retrieval_response = retrieval.retrieve(user_input, 3)
        response = llm.generate_response(user_input, retrieval_response)
        for chunk in response:
            if chunk is not None:
                print(chunk, end="", flush=True)
        print()


def main():
    print("=== Local RAG ===")
    print_help()
    config = Config()
    print("Loading models...")
    retrieval = Retrieval(config)
    llm = LLM(config, "You are a helpful assistant")
    print("Ready.\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd == "generate":
            run_generate(config)
        elif cmd == "query":
            run_query(retrieval, llm)
        elif cmd == "help":
            print_help()
        elif cmd == "exit":
            print("Goodbye.")
            break
        elif cmd == "":
            continue
        else:
            print(f"Unknown command '{cmd}'. Type 'help' to see available commands.")


if __name__ == "__main__":
    main()
