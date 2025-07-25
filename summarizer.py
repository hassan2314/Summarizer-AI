from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from dotenv import load_dotenv
import os
import requests

app = FastAPI()
load_dotenv()

# === Hugging Face Inference API ===
HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN:", HF_TOKEN)
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def call_hf_api(prompt: str, max_tokens=300):
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        print("HF_TOKEN:", HF_TOKEN)
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"HF API Error: {str(e)}"

def generate_bullets_point(dialogue: str):
    prompt = f"Summarize the following text into 3-4 bullet points:\n\n{dialogue}"
    return call_hf_api(prompt, max_tokens=300)

def generate_tags(dialogue: str):
    prompt = f"Extract 3 relevant tags or keywords from this text:\n\n{dialogue}"

    return call_hf_api(prompt, max_tokens=50)

# === Load Models & Tokenizers (for local summary/question models) ===
summarizer_model_path = "./t5_summarizer_final"
question_model_path = "./t5_question_gen_model"

summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_path)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_path)

question_tokenizer = T5Tokenizer.from_pretrained(question_model_path)
question_model = T5ForConditionalGeneration.from_pretrained(question_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer_model.to(device)
question_model.to(device)

# === Request Model ===
class SummaryRequest(BaseModel):
    dialogue: str
    mode: str  # "paragraph", "bullets", or "questions"

# === Main Endpoint ===
@app.post("/summarize")
def summarize(request: SummaryRequest):
    dialogue = request.dialogue.strip()
    tags = generate_tags(dialogue)

    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result, "tags": tags}

    # === Local Models Config ===
    gen_config = {
        "max_length": 1024,
        "num_beams": 9,
        "repetition_penalty": 2.0,
        "temperature": 1.5,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }

    # === Select Prompt and Model ===
    if request.mode == "paragraph":
        prompt = "summarize: "
        model = summarizer_model
        tokenizer = summarizer_tokenizer

    elif request.mode == "questions":
        prompt = (
            "From the following article, generate exactly 3 unique and insightful questions. "
            "Avoid repeating any information. Do not include any statements. "
            "Only include well-formed questions:\n\n"
        )
        model = question_model
        tokenizer = question_tokenizer

    # === Tokenize and Generate ===
    input_text = prompt + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if request.mode == "questions":
        questions = re.split(r"\n+|\d+\.\s+|•\s*", decoded_output)
        questions = [q.strip() for q in questions if q.strip()]
        return {"data": questions, "tags": tags}

    return {"data": decoded_output, "tags": tags}
