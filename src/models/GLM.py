import random
import time

from zhipuai import ZhipuAI
from .Model import Model

class GLM(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config["model_info"]["name"]
        self.temperature = float(config["params"].get("temperature", 0.7))
        self.max_output_tokens = int(config["params"].get("max_output_tokens", 1500))
        self.thinking = config["params"].get("thinking", None)
        self.timeout = float(config["params"].get("timeout", 180.0))
        self.max_retries = max(int(config["params"].get("max_retries", 2)), 0)
        self.retry_backoff = float(config["params"].get("retry_backoff", 2.0))
        self.max_retry_sleep = float(config["params"].get("max_retry_sleep", 30.0))

        api_pos = int(config["api_key_info"]["api_key_use"])
        api_key = config["api_key_info"]["api_keys"][api_pos]
        base_url = config.get("model_info", {}).get("base_url", None)

        if base_url:
            self.client = ZhipuAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        else:
            self.client = ZhipuAI(
                api_key=api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

    def query(self, msg: str) -> str:
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": msg}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_output_tokens,
                    "stream": False,
                    "timeout": self.timeout,
                }
                if self.thinking:
                    kwargs["thinking"] = self.thinking

                response = self.client.chat.completions.create(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return content
                last_error = "empty response"
            except Exception as e:
                last_error = e
                error_text = str(e)
                if any(
                    marker in error_text
                    for marker in [
                        "\u4f59\u989d\u4e0d\u8db3",
                        "\u65e0\u53ef\u7528\u8d44\u6e90\u5305",
                        "invalid api key",
                        "invalid_api_key",
                        "unauthorized",
                    ]
                ):
                    return f"GLM API Error: {last_error}"

            if attempt < self.max_retries:
                sleep_seconds = min(self.max_retry_sleep, self.retry_backoff * (2 ** attempt))
                sleep_seconds += random.uniform(0.0, 0.5)
                time.sleep(sleep_seconds)

        return f"GLM API Error: {last_error}"
