MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

CONFLICT_PROMPT = 'You are a helpful assistant. Below is a query and some context documents. \
The documents may contain conflicting information. \
Please answer the question. If you find contradictions between the documents, explicitly state them and provide details for each viewpoint. \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'


MULTI_VIEWPOINT_PROMPT = 'You are a helpful assistant. Below is a query and multiple groups of context documents representing different viewpoints. \
The documents have been grouped by their conflicting information. \
Please provide a separate answer for EACH viewpoint group to help the user understand the different perspectives. \
\n\nViewpoints:\n[context] \n\nQuery: [question] \n\nAnswer:'


def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    elif prompt_id == 5:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = CONFLICT_PROMPT.replace('[question]', question).replace('[context]', context_str)
    elif prompt_id == 6:
        # context is expected to be a list of strings, where each string is a formatted group
        assert type(context) == list
        context_str = "\n\n".join(context)
        input_prompt = MULTI_VIEWPOINT_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

