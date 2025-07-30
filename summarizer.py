from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List
from functools import lru_cache

# Initialize app and load environment
app = FastAPI(title="Text Processing API", version="1.0")
load_dotenv()

# === Configuration ===
GOOGLE_API_KEY = os.getenv("GEMINI_API")
if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"

# === Models ===
class QAItem(BaseModel):
    question: str
    user_answer: str

class BatchAnswerCheckRequest(BaseModel):
    context: str
    qa_pairs: List[QAItem]

class SummaryRequest(BaseModel):
    dialogue: str
    mode: str  # "paragraph", "bullets", or "questions"

# === Cached Resources === 
@lru_cache(maxsize=1)
def get_summarizer_model():
    model_path = "./t5_summarizer_final"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

# === Helper Functions ===
def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini model with error handling."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

def clean_tags(raw_text: str) -> str:
    """Clean generated tags by removing prefix phrases."""
    return re.sub(r"(?i)^here are.*?:", "", raw_text).strip()

# === API Endpoints ===
@app.post("/check-batch", summary="Check correctness of batch answers")
async def check_batch_answers(request: BatchAnswerCheckRequest):
    """
    Evaluate correctness of user answers against a given context.
    Returns feedback in format: Q1: Correct/Incorrect, Q2: Correct/Incorrect, etc.
    """
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
    """
    
    feedback = get_gemini_response(prompt)
    return {"feedback": feedback}

@app.post("/summarize", summary="Generate summary in different formats")
async def summarize(request: SummaryRequest):
    """
    Generate summaries in different formats:
    - 'paragraph': Traditional summary
    - 'bullets': Bullet point summary
    - 'questions': Generate questions about the text
    Returns summary and generated tags.
    """
    dialogue = request.dialogue.strip()
    tags = generate_tags(dialogue)
    
    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result, "tags": tags}
    
    if request.mode == "questions":
        question_result = generate_questions(dialogue)
        return {"data": question_result, "tags": tags}
    
    # Default to paragraph summary
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

# === Text Processing Functions ===
def generate_bullets_point(text: str) -> str:
    """Generate bullet points from text."""
    prompt = f"From this article, generate bullet points without *:\n\n{text}"
    return get_gemini_response(prompt)

def generate_tags(text: str) -> str:
    """Generate tags from text."""
    prompt = f"From this article, generate 3 tags (just tags, no explanation):\n\n{text}"
    raw_text = get_gemini_response(prompt)
    return clean_tags(raw_text)

def generate_questions(text: str) -> str:
    """Generate questions from text."""
    prompt = f"From this text, generate 3 simple unique questions:\n\n{text}"
    return get_gemini_response(prompt)