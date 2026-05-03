import json

def load_json(file_path):
    with open(file_path,encoding="utf-8-sig") as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'gpt':
        from .GPT import GPT
        model = GPT(config)
    elif provider == 'llama':
        from .Llama import Llama
        model = Llama(config)
    elif provider == 'deepseek':
        from .DeepSeek import DeepSeek
        model = DeepSeek(config)
    elif provider == 'glm':
        from .GLM import GLM
        model = GLM(config)
    else:
        raise ValueError(
            f"ERROR: Unsupported provider '{provider}' in demo release. "
            "Use one of: gpt, llama, deepseek, glm."
        )
    return model
