from config.llm_config import llm
from agents.leave_me_agent import run_leave_me_agent
from agents.user_parser_agent import run_user_parser_agent

def main():
    print("Running REMAS: Leave Me Agent Example")
    user_input = input("Type your message: ")

    response = run_leave_me_agent(user_input=user_input, llm=llm)
    print("Agent Response:", response)
    

if __name__ == "__main__":
    main()
