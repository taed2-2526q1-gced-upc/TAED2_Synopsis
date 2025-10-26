import os
import json
from pathlib import Path
from typing import List
from datetime import datetime
from codecarbon import EmissionsTracker
from transformers import pipeline, AutoTokenizer
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/bielupc/TAED2_Synopsis.mlflow"
)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

EMISSIONS_REPORT_PATH = Path("reports/emissions_report.json")


def _load_report():
    """Load existing emissions report or create new one."""
    if EMISSIONS_REPORT_PATH.exists():
        with open(EMISSIONS_REPORT_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "project": "bart_news_summarization",
        "total_kg": 0.0,
        "total_g": 0.0,
        "records": []
    }


def _save_report(report):
    """Save emissions report to JSON file."""
    EMISSIONS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMISSIONS_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def log_emissions_report(stage: str, emissions_kg: float):
    """Log CO2 emissions to a local JSON report."""
    emissions_kg = emissions_kg or 0.0
    report = _load_report()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stage": stage,
        "emissions_kg": emissions_kg,
        "emissions_g": emissions_kg * 1000,
    }
    report["records"].append(entry)
    report["total_kg"] += emissions_kg
    report["total_g"] += emissions_kg * 1000
    _save_report(report)


def _split_text(text: str, tokenizer, max_tokens: int = 900) -> List[str]:
    """Split text into chunks at sentence boundaries, limited by max_tokens."""
    sentences = (
        text.replace("! ", "!|")
        .replace("? ", "?|")
        .replace(". ", ".|")
        .split("|")
    )
    chunks, current_chunk, current_tokens = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_tokens + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = len(sentence_tokens)
        else:
            current_chunk.append(sentence)
            current_tokens += len(sentence_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def _compute_summary_lengths(token_count: int):
    """Compute dynamic summary min/max lengths based on token count."""
    max_len = min(128, max(30, int(token_count * 0.15)))
    min_len = max(10, int(max_len * 0.4))
    return min_len, max_len


def _summarize_text(
    summarizer,
    text: str,
    tokenizer,
    max_input_length: int = 1024,
):
    """Summarize a possibly long text by splitting into chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(tokens) <= max_input_length:
        min_len, max_len = _compute_summary_lengths(len(tokens))
        result = summarizer(
            text, max_length=max_len, min_length=min_len, do_sample=False
        )
        return result[0]["summary_text"]

    chunks = _split_text(text, tokenizer, max_tokens=900)
    print(f"Text is long ({len(tokens)} tokens). Splitting into {len(chunks)} chunks.")

    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=True))
        min_len, max_len = _compute_summary_lengths(chunk_tokens)
        print(f"Processing chunk {i}/{len(chunks)}...")
        result = summarizer(
            chunk, max_length=max_len, min_length=min_len, do_sample=False
        )
        chunk_summaries.append(result[0]["summary_text"])

    combined_text = " ".join(chunk_summaries)
    combined_tokens = tokenizer.encode(combined_text, add_special_tokens=True)
    min_len, max_len = _compute_summary_lengths(len(combined_tokens))

    if len(combined_tokens) > max_input_length:
        print("Creating final summary from combined chunks...")
        final_result = summarizer(
            combined_text, max_length=max_len, min_length=min_len, do_sample=False
        )
        return final_result[0]["summary_text"]

    return combined_text


def predict_model(
    text: str,
    mlflow_artifact_uri: str = (
        "mlflow-artifacts:/9b4993739fa24d1b80a3e605434b618d/"
        "98108604288f454f89806e859eaf71f8/artifacts/bart-large-cnn-finetuned"
    ),
) -> str:
    """
    Generate a summary for the input text.

    Handles long texts by chunking and adjusts summary length dynamically.
    Tracks CO2 emissions during inference.
    """
    tracker = EmissionsTracker(
        project_name="bart_news_summarization_inference",
        output_dir="reports",
        log_level="warning",
    )
    tracker.start()

    try:
        if not text or not text.strip():
            return ""

        local_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=mlflow_artifact_uri
        )

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        summarizer = pipeline("summarization", model=local_model_path)

        summary = _summarize_text(summarizer, text, tokenizer)
        return summary

    finally:
        emissions_kg = tracker.stop()
        log_emissions_report("inference", emissions_kg)
