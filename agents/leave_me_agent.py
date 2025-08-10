from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from utils.logger import init_log_file, log_token_usage

DEFAULT_PROMPT = "You must always respond with: Leave me alone!"

def run_leave_me_agent(user_input: str, llm, log_file: str = "logs/token_usage.csv") -> str:
    init_log_file(log_file)

    messages = [
        SystemMessage(content=DEFAULT_PROMPT),
        HumanMessage(content=user_input)
    ]

    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        log_token_usage(log_file, cb, user_input)

    return response.content