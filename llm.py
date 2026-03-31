from openai import OpenAI
from sklearn import base
from config import Config


class LLM:

    def __init__(self, config: Config, system_prompt: str = "You are a helpful assistant") -> None:
        self.config = config
        self.client = OpenAI(api_key="",
                             base_url="http://localhost:11434/v1/")
        self.system_prompt = system_prompt
        self.history = [{'role': "system", "content": system_prompt}]

    def generate_response(self, query: str):
        self.history.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.config.local_model, messages=self.history, stream=True)  # type: ignore
        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                result += content
            yield result

        self.history.append({"role": "assistant", "content": result})

    def clear(self):
        self.history = [{'role': "system", "content": self.system_prompt}]
