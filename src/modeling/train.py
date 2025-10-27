import time
import json
from pathlib import Path
from datetime import datetime

import mlflow
import numpy as np
from codecarbon import EmissionsTracker
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import torch

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
        "emissions_g": emissions_kg * 1000
    }
    report["records"].append(entry)
    report["total_kg"] += emissions_kg
    report["total_g"] += emissions_kg * 1000
    _save_report(report)


def preprocess_function(examples, tokenizer, max_input_length=512,
                        max_target_length=128):
    """Preprocess with truncation for speed."""
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    labels = tokenizer(
        examples["highlights"],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge_metric):
    """Compute ROUGE metrics for evaluation."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v * 100, 4) for k, v in result.items()}


def load_and_prepare_data(params, tokenizer):
    """Load dataset and preprocess it."""
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/raw/train.parquet",
            "validation": "data/raw/validation.parquet",
            "test": "data/raw/test.parquet",
        }
    )
    for split in ["train", "validation", "test"]:
        sample_size = min(params["sample_size"], len(dataset[split]))
        dataset[split] = dataset[split].select(range(sample_size))

    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_dataset


def main():
    """Main training function."""
    tracker = EmissionsTracker(
        project_name="bart_news_summarization",
        output_dir="reports",
        log_level="warning"
    )
    tracker.start()

    params = {
        "model_name": "facebook/bart-large-cnn",
        "learning_rate": 3e-5,
        "batch_size": 1,
        "num_epochs": 4,
        "sample_size": 1000,
        "gradient_accumulation_steps": 2,
    }

    with mlflow.start_run(run_name="bart_finetuning"):
        mlflow.log_params(params)

        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(params["model_name"])
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        tokenized_dataset = load_and_prepare_data(params, tokenizer)

        rouge_metric = evaluate.load("rouge")
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir="models/checkpoints",
            eval_strategy="epoch",
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"],
            num_train_epochs=params["num_epochs"],
            save_total_limit=1,
            logging_steps=10,
            save_steps=10000,
            report_to=["mlflow"],
            predict_with_generate=True,
            fp16=False,
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            generation_max_length=128,
            generation_num_beams=4,
            max_grad_norm=1.0,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred, tokenizer, rouge_metric
            ),
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        train_start = time.time()
        train_result = trainer.train()
        train_time = time.time() - train_start

        mlflow.log_metrics({
            "train_loss": getattr(train_result, "training_loss", 0.0),
            "train_time_minutes": train_time / 60,
        })

        eval_results = trainer.evaluate(
            eval_dataset=tokenized_dataset["test"],
            metric_key_prefix="test"
        )
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        model_path = Path("models") / "bart-large-cnn-finetuned"
        model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        mlflow.log_artifacts(str(model_path), artifact_path="bart-large-cnn-finetuned")

        training_emissions_kg = tracker.stop()
        log_emissions_report("training", training_emissions_kg)


if __name__ == "__main__":
    main()
