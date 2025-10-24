from pathlib import Path
from typing import List
from transformers import pipeline, AutoTokenizer

def summarizer(text: str, model_path: str = "models/bart-large-cnn-finetuned") -> str:
    """
    Generate a summary for the input text. Handles long texts by chunking.
    Dynamic min/max summary lengths based on input size.
    """
    if not text or not text.strip():
        return ""

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}, using pretrained facebook/bart-large-cnn")
        model_path = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.encode(text, add_special_tokens=True)
    max_input_length = 1024

    def compute_summary_lengths(token_count):
        max_len = min(128, max(30, int(token_count * 0.15)))
        min_len = max(10, int(max_len * 0.4))
        return min_len, max_len

    min_len, max_len = compute_summary_lengths(len(tokens))

    if len(tokens) <= max_input_length:
        summarizer = pipeline("summarization", model=model_path)
        result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]['summary_text']

    chunks = _split_text(text, tokenizer, max_tokens=900)
    print(f"Text is long ({len(tokens)} tokens). Splitting into {len(chunks)} chunks.")

    summarizer = pipeline("summarization", model=model_path)
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=True))
        min_len, max_len = compute_summary_lengths(chunk_tokens)
        print(f"Processing chunk {i}/{len(chunks)}...")
        result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        chunk_summaries.append(result[0]['summary_text'])

    combined_text = ' '.join(chunk_summaries)
    combined_tokens = tokenizer.encode(combined_text, add_special_tokens=True)
    min_len, max_len = compute_summary_lengths(len(combined_tokens))

    if len(combined_tokens) > max_input_length:
        print("Creating final summary from combined chunks...")
        final_result = summarizer(combined_text, max_length=max_len, min_length=min_len, do_sample=False)
        return final_result[0]['summary_text']

    return combined_text


def _split_text(text: str, tokenizer, max_tokens: int = 900) -> List[str]:
    """
    Split text into chunks at sentence boundaries, limited by max_tokens.
    """
    sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
    chunks, current_chunk, current_tokens = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_tokens + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = len(sentence_tokens)
        else:
            current_chunk.append(sentence)
            current_tokens += len(sentence_tokens)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text]
