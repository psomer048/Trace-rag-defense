import time

from openai import OpenAI
import openai  # Translated comment (English only).
from .Model import Model

class DeepSeek(Model):
    def __init__(self, config):
        super().__init__(config)

        self.model_name = config["model_info"]["name"]
        self.temperature = float(config["params"].get("temperature", 0.7))
        self.max_output_tokens = int(config["params"].get("max_output_tokens", 1500))

        # :/( config )
        self.timeout = float(config["params"].get("timeout", 180.0))      # 180 (3)
        self.max_retries = int(config["params"].get("max_retries", 2))
        self.retry_backoff = float(config["params"].get("retry_backoff", 2.0))

        api_pos = int(config["api_key_info"]["api_key_use"])
        api_key = config["api_key_info"]["api_keys"][api_pos]

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=self.timeout,          # Translated comment (English only).
            max_retries=self.max_retries,  # Translated comment (English only).
        )

    def query(self, msg: str) -> str:
        client = self.client.with_options(timeout=self.timeout, max_retries=0)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": msg}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    stream=False,
                )
                return completion.choices[0].message.content.strip()

            except openai.APITimeoutError as e:
                last_error = f"timeout: {e}"
            except openai.APIConnectionError as e:
                last_error = f"connection error: {e}"
            except openai.APIStatusError as e:
                status_code = getattr(e, "status_code", None)
                body = getattr(e, "body", None)
                if status_code == 402:
                    return f"DeepSeek API Error: status 402: {body}"
                if status_code == 429:
                    last_error = f"rate limit: {body}"
                else:
                    return f"DeepSeek API Error: status {status_code}: {body}"
            except Exception as e:
                return f"DeepSeek API Error: {e}"

            if attempt < self.max_retries:
                sleep_seconds = self.retry_backoff * (2 ** attempt)
                time.sleep(sleep_seconds)

        return f"DeepSeek API Error: {last_error or 'unknown error'}"
