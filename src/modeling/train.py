import os
import time
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
from codecarbon import EmissionsTracker
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=64):
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        examples["highlights"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge_metric):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}


def log_emissions_report(tracker, emissions):
    if emissions is None:
        emissions = 0.0

    emissions_data = {
        "co2_emissions_kg": emissions,
        "co2_emissions_g": emissions * 1000
    }
    mlflow.log_metrics(emissions_data)


def main(params_override=None):
    tracker = EmissionsTracker(project_name="bart_news_summarization", output_dir="reports", log_level="warning")
    tracker.start()

    params = {
        "model_name": "facebook/bart-large-cnn",
        "learning_rate": 5e-5,
        "batch_size": 4,
        "num_epochs": 1,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_input_length": 128,
        "max_target_length": 32
    }

    with mlflow.start_run(run_name="bart_news_training"):
        mlflow.log_params(params)

        dataset = load_from_disk("data/raw")
        # small dataset
        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].select(range(min(5, len(dataset[split])))) 

        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(params['model_name'])

        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                params['max_input_length'],
                params['max_target_length']
            ),
            batched=True,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False
        )

        rouge_metric = evaluate.load("rouge")
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir="models/checkpoints",
            eval_strategy="steps",
            eval_steps=500,
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['batch_size'],
            per_device_eval_batch_size=params['batch_size'],
            weight_decay=params['weight_decay'],
            save_total_limit=3,
            num_train_epochs=params['num_epochs'],
            predict_with_generate=True,
            logging_steps=100,
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            warmup_steps=params['warmup_steps'],
            report_to=["mlflow"]
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, rouge_metric)
        )

        train_start = time.time()
        train_result = trainer.train()
        train_time = time.time() - train_start

        mlflow.log_metrics({
            "train_loss": getattr(train_result, "training_loss", 0.0),
            "train_time_hours": train_time / 3600,
        })

        eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"test_{key}", value)

        model_path = Path("models") / "bart-large-cnn-finetuned"
        model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="bart-large-cnn-finetuned",
            task="summarization"
        )

        emissions = tracker.stop()
        log_emissions_report(tracker, emissions)


if __name__ == "__main__":
    main()
