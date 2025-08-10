from config.embedding_model_config import embeddings as client
from openai._exceptions import RateLimitError
import backoff
from typing import Union, List
from utils.logger import log_embedding_token_usage

# Your Azure deployment name (must match what you set in Azure Portal)
deployment_name = "team8-embedding"

# ------------- Retry Decorators -------------

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def embed_single_text(text: str) -> List[float]:
    """
    Embed a single string using Azure OpenAI embedding deployment.
    """
    response = client.embeddings.create(
        input=[text],
        model=deployment_name
    )
    return response.data[0].embedding

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def embed_batch_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Azure OpenAI in batch.
    """
    response = client.embeddings.create(
        input=texts,
        model=deployment_name
    )
    return [item.embedding for item in response.data]

# ------------- Unified Interface -------------

def embed_texts(input_text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Main function: Takes a string or list of strings and returns embeddings.
    Automatically selects between single and batch mode.
    Also logs token usage to 'logs/embedding_counts.csv'.
    """
    log_file = "logs/embedding_counts.csv"

    if isinstance(input_text, str):
        # Log tokens for single input
        result = embed_single_text(input_text)
        log_embedding_token_usage(log_file, input_text)
        return result

    elif isinstance(input_text, list):
        result = embed_batch_texts(input_text)
        for text in input_text:
            log_embedding_token_usage(log_file, text)
        return result

    else:
        raise ValueError("Input must be a string or list of strings.")