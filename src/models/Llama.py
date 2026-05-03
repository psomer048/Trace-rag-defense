import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .Model import Model

class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.temperature = float(config["params"].get("temperature", 0.1))

        # stop phrase,
        self.stop_phrases = ["$"]   # stop phrase

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        model_config = AutoConfig.from_pretrained(self.name, token=hf_token)

        # LLaMA3 rope_scaling
        if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
            rope = model_config.rope_scaling
            model_config.rope_scaling = {
                "type": rope.get("rope_type", "rope"),
                "factor": rope.get("factor", 1.0)
            }

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=hf_token)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            config=model_config,
            torch_dtype=torch.float16,
            device_map={"": 0}
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_length = getattr(self.model.config, 'max_position_embeddings', 4096)

        # stop phrase token id
        self.stop_ids = [
            self.tokenizer.convert_tokens_to_ids(sp) if sp in self.tokenizer.get_vocab()
            else self.tokenizer.encode(sp, add_special_tokens=False)[-1]
            for sp in self.stop_phrases
        ]

    def _apply_stop(self, text):
        """
        Truncate output at the first configured stop phrase.
        """
        for sp in self.stop_phrases:
            idx = text.find(sp)
            if idx != -1:
                return text[:idx]
        return text

    def query(self, msg):
        try:
            device = next(self.model.parameters()).device

            # LLaMA3 chat
            messages = [
               {"role": "user", "content": msg}
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_output_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature,
                    repetition_penalty=1.1,   # Translated comment (English only).
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            result = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

            # stop phrase()
            result = self._apply_stop(result)

            return result.strip()

        except Exception as e:
            print(f"Error in Llama query: {e}")
            return f"Error: Unable to generate response due to: {str(e)}"

