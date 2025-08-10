import os
import csv
from datetime import datetime
import tiktoken 

# ------------------ Init Logger ------------------

def init_log_file(log_file: str):
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "input_tokens", "output_tokens", "total_tokens", "user_input"])




# ------------------ Log Chat Token Usage ------------------

def log_token_usage(log_file: str, cb, user_input: str):
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_tokens,
            user_input
        ])

# ------------------ Count Embedding Tokens ------------------

def count_embedding_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """
    Returns the number of tokens in the input text for the embedding model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# ------------------ Count Embedding Tokens ------------------

def count_embedding_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """
    Returns the number of tokens in the input text for the specified embedding model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback in case the model is not recognized
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ------------------ Init Embedding Token Log File ------------------

def init_embedding_log_file(log_file: str):
    """
    Initializes the embedding token usage log file with headers.
    """
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "input_tokens", "output_tokens", "total_tokens", "text_input"])

# ------------------ Log Embedding Token Usage ------------------

def log_embedding_token_usage(log_file: str, text: str, model: str = "text-embedding-3-small"):
    """
    Logs embedding token usage to a CSV file.
    """
    init_embedding_log_file(log_file)
    tokens = count_embedding_tokens(text, model)

    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            tokens,
            0,       # output_tokens = 0 for embeddings
            tokens,  # total_tokens = input + output
            text
        ])
