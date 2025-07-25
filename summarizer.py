from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import requests  # Replaced google.generativeai
from dotenv import load_dotenv
import os
from typing import List

app = FastAPI()
load_dotenv()

# === Hugging Face Configuration ===
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def generate_bullets_point(dialogue: str) -> str:
    """Generate bullet points using HF API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        payload = {
            "inputs": f"Summarize this as bullet points:\n{dialogue}",
            "parameters": {"max_length": 200}
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()[0]['summary_text']
    except Exception as e:
        return f"Error generating bullets: {str(e)}"

def generate_tags(dialogue: str) -> List[str]:
    """Generate tags using HF API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/fabiochiu/t5-base-tag-generation"
        payload = {
            "inputs": f"Generate tags for: {dialogue}",
            "parameters": {"num_return_sequences": 3}
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return [tag.strip() for tag in response.json()[0]['generated_text'].split(",")]
    except Exception as e:
        return [f"Error generating tags: {str(e)}"]

# === (Keep your existing T5 model loading code) ===
summarizer_model_path = "./t5_summarizer_final"
question_model_path = "./t5_question_gen_model"

summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_path)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_path)

question_tokenizer = T5Tokenizer.from_pretrained(question_model_path)
question_model = T5ForConditionalGeneration.from_pretrained(question_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer_model.to(device)
question_model.to(device)

# === (Keep your existing Request Model and Main Endpoint) ===
class SummaryRequest(BaseModel):
    dialogue: str
    mode: str  # "paragraph", "bullets", or "questions"

@app.post("/summarize")
def summarize(request: SummaryRequest):
    dialogue = request.dialogue.strip()
    tags = generate_tags(dialogue)
    
    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result, "tags": tags}

    # ... (rest of your existing summarize function remains the same)