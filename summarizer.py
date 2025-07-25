from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

# === Gemini Configuration ===
GOOGLE_API_KEY = os.getenv("GEMINI_API")
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

def generate_tags(dialogue: str):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"From the following article, generate 3 tags (just tags, no explanation):\n\n{dialogue}"
        )
        # Remove any unwanted prefix using regex or string manipulation
        raw_text = response.text.strip()

        # Optional cleanup (if model still returns numbered or prefixed tags)
        cleaned_tags = re.sub(r"(?i)^here are.*?:", "", raw_text).strip()
        return cleaned_tags
    except Exception as e:
        return f"Error generating tags: {str(e)}"

def generate_questions(dialogue: str):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use correct model name
        response = model.generate_content(
            f"From the following text, generate  3 simple unique questions: \n\n{dialogue}"
        )
        return response.text
    except Exception as e:
        return f"Error generating questions: {str(e)}"

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
    tags=generate_tags(dialogue)
    
    if request.mode == "bullets":
        bullet_result = generate_bullets_point(dialogue)
        return {"data": bullet_result, "tags": tags}

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
        question_result = generate_questions(dialogue)
        return {"data": question_result, "tags": tags}

    

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
        return {"data": questions, "tags": tags}

    # === Return Paragraph Summary ===
    return {"data": decoded_output, "tags": tags}
