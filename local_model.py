from deepeval.models.base_model import DeepEvalBaseLLM
import json, requests

class LocalModel(DeepEvalBaseLLM):
    def __init__(self, base_url="http://localhost:8080/v1", model="gemma3", api_key="local-key"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _chat(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return self._chat(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Local({self.model})"
