from transformers import pipeline

analyzer = pipeline("text-classification", model="boltuix/bert-emotion", return_all_scores=True)