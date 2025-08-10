from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = "8HIrkiHT0KYgwTOXUYb7PjXnQAo7x7lk6mxy0BXSQAh9hNbpVOVqJQQJ99AKACYeBjFXJ3w3AAABACOG7iAn"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"



embeddings = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
