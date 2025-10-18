"""
Train model for news summarization.
"""
import os
import sys
import time
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, tokenizer, rouge_metric):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    metrics = {
        "rouge1": round(result["rouge1"] * 100, 4),
        "rouge2": round(result["rouge2"] * 100, 4),
        "rougeL": round(result["rougeL"] * 100, 4),
    }
    
    return metrics


def main():
    """Main training function with robust error handling."""
    try:
        tracker = EmissionsTracker(
            project_name="bart_news_summarization",
            output_dir="reports",
            log_level="warning"
        )
        tracker.start()
        
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        params = {
            "model_name": "facebook/bart-large-cnn",
            "learning_rate": 2e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "num_epochs": 5,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        }
        
        with mlflow.start_run(run_name="bart_news_training"):
            mlflow.log_params(params)
            
            logger.info("Loading dataset from data/processed")
            data_path = Path("data/raw")
            
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {data_path}. Run preprocessing first."
                )
            
            dataset = load_from_disk(str(data_path))
            
            required_splits = ["train", "validation", "test"]
            for split in required_splits:
                if split not in dataset:
                    raise ValueError(f"Missing required split: {split}")
                logger.info(f"{split} split: {len(dataset[split])} samples")
            
            logger.info(f"Loading model: {params['model_name']}")
            tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
            model = AutoModelForSeq2SeqLM.from_pretrained(params['model_name'])
            
            rouge_metric = load_metric("rouge")
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            
            logger.info("Configuring training arguments")
            training_args = Seq2SeqTrainingArguments(
                output_dir="models/checkpoints",
                evaluation_strategy="steps",
                eval_steps=500,
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=params['batch_size'],
                per_device_eval_batch_size=params['batch_size'],
                gradient_accumulation_steps=params['gradient_accumulation_steps'],
                weight_decay=params['weight_decay'],
                max_grad_norm=params['max_grad_norm'],
                save_total_limit=3,
                num_train_epochs=params['num_epochs'],
                predict_with_generate=True,
                logging_steps=100,
                save_steps=500,
                load_best_model_at_end=True,
                metric_for_best_model="rouge1",
                greater_is_better=True,
                warmup_steps=params['warmup_steps'],
                lr_scheduler_type=params['lr_scheduler_type'],
                fp16=True,
                dataloader_num_workers=4,
                report_to=["mlflow"],
                seed=42,
            )

            logger.info("Initializing trainer")
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda eval_pred: compute_metrics(
                    eval_pred, tokenizer, rouge_metric
                )
            )
            
            logger.info("Starting training")
            train_start = time.time()
            train_result = trainer.train()
            train_time = time.time() - train_start
            
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_time_hours": train_time / 3600,
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            })
            
            logger.info("Evaluating on test set")
            eval_results = trainer.evaluate(eval_dataset=dataset["test"])
            
            mlflow.log_metrics({
                "test_rouge1": eval_results.get("eval_rouge1", 0),
                "test_rouge2": eval_results.get("eval_rouge2", 0),
                "test_rougeL": eval_results.get("eval_rougeL", 0),
                "test_loss": eval_results.get("eval_loss", 0),
            })
            
            logger.info("Saving model")
            model_path = Path("models") / "bart_news_final"
            model_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                artifact_path="model",
                task="summarization"
            )
            
            emissions = tracker.stop()
            mlflow.log_metrics({
                "co2_emissions_g": emissions * 1000,
                "energy_kwh": tracker.final_emissions_data.energy_consumed,
            })
            
            logger.info("Training completed successfully")
            print(f"\nTraining Results:")
            print(f"ROUGE-1: {eval_results.get('eval_rouge1', 0):.2f}")
            print(f"ROUGE-2: {eval_results.get('eval_rouge2', 0):.2f}")
            print(f"ROUGE-L: {eval_results.get('eval_rougeL', 0):.2f}")
            print(f"Test Loss: {eval_results.get('eval_loss', 0):.4f}")
            print(f"CO2: {emissions * 1000:.2f}g")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
