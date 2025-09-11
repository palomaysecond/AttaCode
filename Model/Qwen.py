import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

class Qwen:
    def __init__(self, api_key: str, model_name: str, base_url: str = "<API_BASE_URL>"):
        openai.api_key = api_key
        openai.api_base = base_url
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
