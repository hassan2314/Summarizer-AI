from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import google.generativeai as genai
import os

app = FastAPI()

# === Gemini Configuration ===
GOOGLE_API_KEY = "AIzaSyAojwXjKCD19p9I5ObQfelBS5_ECfzy7Dg"
genai.configure(api_key=GOOGLE_API_KEY)

def generate_bullets_point(dialogue: str):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use correct model name
        response = model.generate_content(
            f"From the following article, generate bullet points:\n\n{dialogue}"
        )
        return response.text
    except Exception as e:
        return f"Error generating bullets: {str(e)}"

# === Load Models & Tokenizers ===
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

    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result}

    # === Generation Config ===
    gen_config = {
        "max_length": 1024,
        "num_beams": 9,
        "repetition_penalty": 2.0,
        "temperature": 1.5,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }

    # === Select Model and Prompt ===
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
    else:
        prompt = "summarize: "
        model = summarizer_model
        tokenizer = summarizer_tokenizer

    # === Tokenize and Generate ===
    input_text = prompt + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # === Post-process Questions ===
    if request.mode == "questions":
        questions = re.split(r"\n+|\d+\.\s+|•\s*", decoded_output)
        questions = [q.strip() for q in questions if q.strip()]
        return {"data": questions}

    # === Return Paragraph Summary ===
    return {"data": decoded_output}
