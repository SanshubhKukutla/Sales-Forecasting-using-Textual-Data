import pandas as pd
from transformers import AutoTokenizer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(reviews, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(reviews.tolist(), return_tensors="pt", padding=True, truncation=True)
    return inputs

