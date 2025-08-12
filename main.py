from config.llm_config import llm
from agents.user_parser_agent import invoke_user_parser_agent

def main():
    print("Running REMAS:")
    user_input = input("Type your message: ")
    res = invoke_user_parser_agent(user_input)
    print("Response from user parser agent:", res)
    

if __name__ == "__main__":
    main()
