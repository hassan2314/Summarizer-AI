from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import httpx
import os
from typing import List
from functools import lru_cache
from dotenv import load_dotenv

# === App Setup ===
app = FastAPI(title="Text Processing API", version="1.0")
load_dotenv()

# === Config ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env")


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

# === Caching T5 Local Model ===
@lru_cache(maxsize=1)
def get_summarizer_model():
    model_path = "./t5_summarizer_final"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

# === Groq Helper ===
def get_groq_response(prompt: str) -> str:
    """Send prompt to Groq LLaMA-3 and return response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

# === Cleaning Utility ===
def clean_tags(raw_text: str) -> str:
    return re.sub(r"(?i)^here are.*?:", "", raw_text).strip()

# === Summarization API ===
@app.post("/summarize", summary="Generate summary in different formats")
async def summarize(request: SummaryRequest):
    dialogue = request.dialogue.strip()
    tags = generate_tags(dialogue)

    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result, "tags": tags}

    if request.mode == "questions":
        question_result = generate_questions(dialogue)
        return {"data": question_result, "tags": tags}

    # === Paragraph summary from T5 local ===
    tokenizer, model = get_summarizer_model()
    prompt = "summarize: " + dialogue

    gen_config = {
        "max_length": 1024,
        "num_beams": 9,
        "repetition_penalty": 2.0,
        "temperature": 1.5,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"data": decoded_output, "tags": tags}

# === Batch QA Evaluation ===
@app.post("/check-batch", summary="Check correctness of batch answers")
async def check_batch_answers(request: BatchAnswerCheckRequest):
    formatted_qas = "\n".join(
        f"Q{idx+1}: {qa.question}\nA{idx+1}: {qa.user_answer}" 
        for idx, qa in enumerate(request.qa_pairs)
    )

    prompt = f"""
Given the following context:

{request.context}

Evaluate the user's answers to these questions:

{formatted_qas}

Respond in format:
Q1: Correct or Incorrect 
Q2: Correct or Incorrect 
...
without any introductory line, numbering
    """
    feedback = get_groq_response(prompt)
    return {"feedback": feedback}

# === Groq-Backed Text Generators ===
def generate_bullets_point(text: str) -> str:
    prompt = f"""
From the following article, generate bullet points using only plain lines, without any symbols like *, •, -, or numbers. 

Each bullet point should be in a new line, like:
This is the first point  
This is the second point  

Only return the plain text bullet points, nothing else.

Article:
{text}
"""

    return get_groq_response(prompt)

def generate_tags(text: str) -> str:
    prompt = f"From this article, generate 3 tags (just tags, no explanation and no numbers):\n\n{text}"
    raw_text = get_groq_response(prompt)
    return clean_tags(raw_text)

def generate_questions(text: str) -> str:
    prompt = f"""
From the following text, generate exactly 3 simple and unique questions.

Only return the questions directly without any introductory line, numbering, or bullet points.

Each question should be on a new line.

Text:
{text}
"""

    return get_groq_response(prompt)
