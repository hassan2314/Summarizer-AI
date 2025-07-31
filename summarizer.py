from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import httpx
import os
from dotenv import load_dotenv
from typing import List
import google.generativeai as genai

# === App Setup ===
app = FastAPI(title="Text Processing API", version="1.0")
load_dotenv()

# === Load API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API = os.getenv("GEMINI_API")
if not GROQ_API_KEY or not GEMINI_API:
    raise RuntimeError("Missing GROQ_API_KEY or GEMINI_API")

# === Groq Config ===
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# === Gemini Config ===
genai.configure(api_key=GEMINI_API)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
# gemini_model = genai.GenerativeModel("gemma-3-27b-it")

# === Load Local T5 Models ===
summarizer_model_path = "./t5_summarizer_final"
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_path)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer_model.to(device)

# === Request Models ===
class QAItem(BaseModel):
    question: str
    user_answer: str

class BatchAnswerCheckRequest(BaseModel):
    context: str
    qa_pairs: List[QAItem]

class SummaryRequest(BaseModel):
    dialogue: str
    mode: str  # "paragraph", "bullets", or "questions"

# === Groq LLM Call ===
def get_groq_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = httpx.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# === Gemini Answer Check ===
@app.post("/check-batch")
def check_batch_answers(request: BatchAnswerCheckRequest):
    formatted_qas = "\n".join(
        f"Q{idx+1}: {qa.question}\nA{idx+1}: {qa.user_answer}"
        for idx, qa in enumerate(request.qa_pairs)
    )

    prompt = f"""
Given the following context:

{request.context}

Evaluate the user's answers strictly. Only return:

Q1: Correct or Incorrect  
Q2: Correct or Incorrect  
Q3: Correct or Incorrect

{formatted_qas}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return {"feedback": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Summary Endpoint ===
@app.post("/summarize")
def summarize(request: SummaryRequest):
    dialogue = request.dialogue.strip()
    tags = generate_tags(dialogue)

    if request.mode == "bullets":
        return {"data": generate_bullets_point(dialogue), "tags": tags}

    if request.mode == "questions":
        return {"data": generate_questions(dialogue), "tags": tags}

    # === Paragraph Summary via T5 ===
    prompt = "summarize: " + dialogue
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    gen_config = {
        "max_length": 1024,
        "num_beams": 4,
        "repetition_penalty": 2.0,
        "temperature": 1.5,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }

    with torch.no_grad():
        outputs = summarizer_model.generate(**inputs, **gen_config)
    decoded_output = summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"data": decoded_output, "tags": tags}

# === Groq-based Content Generators ===
def generate_bullets_point(text: str) -> str:
    prompt = f"""
From the following article, generate bullet points using plain lines only.

Example format:
This is the first point  
This is the second point  

Do not use bullets, *, -, or numbers. Just one point per line.

Article:
{text}
"""
    return get_groq_response(prompt)

def generate_questions(text: str) -> str:
    prompt = f"""
From the following text, generate exactly 3 simple, unique questions.

Only return the questions directly, no numbers, no bullets.

Text:
{text}
"""
    return get_groq_response(prompt)

def generate_tags(text: str) -> str:
    prompt = f"Generate 3 tags for this text (no numbers, no explanation) seperate with commas:\n\n{text}"
    raw = get_groq_response(prompt)
    return re.sub(r"(?i)^here are.*?:", "", raw).strip()
