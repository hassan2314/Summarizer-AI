# T5 Question Generation & Summarization Models

This repository contains two fine-tuned models based on `t5-small` for:

- ✅ Question Generation
- ✅ Abstractive Summarization

Both models are available on Hugging Face:

- [T5 Question Generator](https://huggingface.co/hassan2314/t5_question_gen_model)
- [T5 Summarizer Final](https://huggingface.co/hassan2314/t5_summarizer_final)

---

## Model 1: `hassan2314/t5_question_gen_model`

### Task: Question Generation

This model takes a declarative sentence or paragraph and generates relevant questions using a prefix like:

```text
generate questions: <your_input_text>
```

# Usage

```
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("hassan2314/t5_question_gen_model")
model = T5ForConditionalGeneration.from_pretrained("hassan2314/t5_question_gen_model")

input_text = "generate questions: The Eiffel Tower is located in Paris."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
