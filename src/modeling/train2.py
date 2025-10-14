"""
This module implements comprehensive quality assurance practices including:
- CO2 emissions tracking
- Energy efficiency monitoring
- MLflow experiment tracking
- Data validation
- Model evaluation metrics
"""
import os
import time
from pathlib import Path
from typing import Dict, Any

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


def validate_dataset(dataset) -> Dict[str, Any]:
    """Validate dataset quality and structure."""
    validation_results = {
        "passed": True,
        "checks": {},
        "warnings": []
    }
    
    required_splits = ['train', 'validation', 'test']
    for split in required_splits:
        if split not in dataset:
            validation_results["passed"] = False
            raise ValueError(f"Missing required split: {split}")
        validation_results["checks"][f"{split}_exists"] = True
    
    min_samples = {'train': 1000, 'validation': 100, 'test': 100}
    for split, min_count in min_samples.items():
        count = len(dataset[split])
        validation_results["checks"][f"{split}_count"] = count
        if count < min_count:
            validation_results["warnings"].append(
                f"{split} has only {count} samples (expected >{min_count})"
            )
    
    required_columns = ['article', 'highlights']
    for split in required_splits:
        cols = dataset[split].column_names
        for col in required_columns:
            if col not in cols:
                validation_results["passed"] = False
                raise ValueError(f"Missing column '{col}' in {split} split")
    
    return validation_results


def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=128):
    """Preprocess dataset for fine-tuning."""
    inputs = examples["article"]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge_metric):
    """Compute ROUGE metrics for evaluation."""
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    result = {key: value * 100 for key, value in result.items()}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def fine_tune_model():
    """
    Fine-tune BART model with comprehensive quality assurance.
    
    Implements:
    - CO2 emissions tracking
    - Data validation
    - Model evaluation with ROUGE metrics
    - Energy efficiency metrics
    - MLflow logging
    """

    tracker = EmissionsTracker(
        project_name="bart_finetuning",
        output_dir="reports/emissions",
        log_level="warning"
    )
    tracker.start()
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    params = {
        "model_name": "facebook/bart-large-cnn",
        "max_input_length": 1024,
        "max_target_length": 128,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 100
    }
    
    with mlflow.start_run(run_name="bart_finetuning_qa"):
        print("=" * 70)
        print("BART Fine-tuning Pipeline - Quality Assurance Enabled")
        print("=" * 70)
        
        mlflow.log_params(params)
        
        print("\n[1/7] Loading and validating dataset...")
        data_path = Path("data/raw")
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Run data preparation first."
            )
        
        dataset = load_from_disk(str(data_path))
        
        print("  → Running data validation checks...")
        validation_results = validate_dataset(dataset)
        
        if validation_results["passed"]:
            print("  ✓ Data validation passed")
        else:
            raise ValueError("Data validation failed")
        
        mlflow.log_dict(validation_results, "data_validation_results.json")
        mlflow.log_metrics({
            "train_samples": validation_results["checks"]["train_count"],
            "validation_samples": validation_results["checks"]["validation_count"],
            "test_samples": validation_results["checks"]["test_count"]
        })
        
        print(f"\n[2/7] Loading model: {params['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(params['model_name'])
        
        print(f"  ✓ Model loaded - Parameters: {model.num_parameters():,}")
        mlflow.log_metric("model_parameters", model.num_parameters())
        
        # Preprocess dataset
        print("\n[3/7] Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                params['max_input_length'],
                params['max_target_length']
            ),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        print("  ✓ Dataset tokenized")
        
        rouge_metric = evaluate.load("rouge")
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )
        
        print("\n[4/7] Configuring training...")
        training_args = Seq2SeqTrainingArguments(
            output_dir="models/bart_finetuned",
            evaluation_strategy="steps",
            eval_steps=params['eval_steps'],
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['batch_size'],
            per_device_eval_batch_size=params['batch_size'],
            weight_decay=params['weight_decay'],
            save_total_limit=3,
            num_train_epochs=params['num_epochs'],
            predict_with_generate=True,
            fp16=False,
            logging_steps=params['logging_steps'],
            save_steps=params['save_steps'],
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
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred, tokenizer, rouge_metric
            )
        )
        
        print("\n[5/7] Starting fine-tuning...")
        train_start = time.time()
        
        train_result = trainer.train()
        
        train_time = time.time() - train_start
        print(f"  ✓ Training completed in {train_time/3600:.2f} hours")
        
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_time_seconds": train_time,
            "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
            "train_steps_per_second": train_result.metrics.get('train_steps_per_second', 0)
        })
        
        print("\n[6/7] Evaluating model on test set...")
        eval_start = time.time()
        
        eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        
        eval_time = time.time() - eval_start
        print(f"  ✓ Evaluation completed in {eval_time:.2f}s")
        
        print("\n  Evaluation Metrics:")
        for key, value in eval_results.items():
            if not key.startswith('eval_runtime'):
                print(f"    • {key}: {value:.4f}")
                mlflow.log_metric(f"test_{key}", value)
        
        print("\n[7/7] Saving fine-tuned model...")
        model_path = "models/bart_finetuned_final"
        os.makedirs(model_path, exist_ok=True)
        
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        mlflow.transformers.log_model(
            transformers_model={
                "model": trainer.model,
                "tokenizer": tokenizer
            },
            artifact_path="finetuned_model",
            task="summarization"
        )
        
        print(f"  ✓ Model saved to {model_path}")
        
        print("\n[Emissions] Collecting environmental impact data...")
        emissions = tracker.stop()
        
        emissions_data = {
            "co2_emissions_kg": emissions,
            "co2_emissions_g": emissions * 1000,
            "energy_consumed_kwh": tracker.final_emissions_data.energy_consumed,
            "duration_seconds": tracker.final_emissions_data.duration
        }
        
        mlflow.log_metrics(emissions_data)
        
        print(f"  → CO2 emissions: {emissions * 1000:.2f}g")
        print(f"  → Energy consumed: {tracker.final_emissions_data.energy_consumed:.6f} kWh")
        
        emissions_file = "reports/emissions/emissions.csv"
        if os.path.exists(emissions_file):
            mlflow.log_artifact(emissions_file, "emissions_report")
        
        print("\n" + "=" * 70)
        print("✓ Fine-tuning completed successfully!")
        print("=" * 70)
        print(f"\nModel Performance:")
        print(f"  • ROUGE-1: {eval_results.get('eval_rouge1', 0):.4f}")
        print(f"  • ROUGE-2: {eval_results.get('eval_rouge2', 0):.4f}")
        print(f"  • ROUGE-L: {eval_results.get('eval_rougeL', 0):.4f}")
        print(f"\nEnvironmental Impact:")
        print(f"  • Total CO2: {emissions * 1000:.2f}g")
        print(f"  • Energy: {tracker.final_emissions_data.energy_consumed:.6f} kWh")
        print(f"\nCheck MLflow UI for detailed metrics and artifacts")


if __name__ == "__main__":
    fine_tune_model()
