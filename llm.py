from openai import OpenAI
from config import Config


class LLM:

    def __init__(self, config: Config, system_prompt: str = "You are a helpful assistant") -> None:
        self.config = config
        self.client = OpenAI(api_key="",
                             base_url="http://localhost:11434/v1/")
        self.system_prompt = system_prompt
        self.history = [{'role': "system", "content": system_prompt}]

    def generate_query(self, query, retrieval_response: list[dict]):

        texts: list[str] = ["\n" + item['text'][:300] for item in retrieval_response]

        return f"""
        Answer below question based on the context given
        
        Question:
        {query}
        
        Context:
         {"".join(texts)}
    """

    def generate_response(self, query: str, retrieval_response: list[dict[str, float]]):
        message_content = self.generate_query(query, retrieval_response)
        print("Generated message content for LLM:")
        print(message_content)
        self.history.append({"role": "user", "content": message_content})

        response = self.client.chat.completions.create(
            model=self.config.local_model, messages=self.history, stream=True)  # type: ignore
        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                result += content
                yield content

        self.history.append({"role": "assistant", "content": result})

    def clear(self):
        self.history = [{'role': "system", "content": self.system_prompt}]
