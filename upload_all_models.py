from huggingface_hub import upload_folder

models = {
    "t5_question_gen_model": "hassan2314/t5_question_gen_model",
    "t5_summarizer_final": "hassan2314/t5_summarizer_final"
}

for folder, repo in models.items():
    print(f"Uploading {folder} to {repo}")
    upload_folder(folder_path=folder, repo_id=repo, repo_type="model")
    print(f"✅ Done: {folder}\n")
