import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("API_KEY")
DEPLOYMENT_NAME = "team8-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=DEPLOYMENT_NAME,
)

# print("LLM configuration initialized with Azure OpenAI settings.")