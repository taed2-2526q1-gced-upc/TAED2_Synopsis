"""
Training script for summarization model using BART with quality assurance.

This module implements comprehensive quality assurance practices including:
- CO2 emissions tracking
- Energy efficiency monitoring
- MLflow experiment tracking
- Data validation
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import mlflow
import mlflow.transformers
import numpy as np
from codecarbon import EmissionsTracker
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)


def validate_dataset(dataset) -> Dict[str, Any]:
    """
    Validate dataset quality and structure.
    
    Args:
        dataset: HuggingFace dataset object
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If dataset fails validation checks
    """
    validation_results = {
        "passed": True,
        "checks": {},
        "warnings": []
    }
    
    # Check dataset splits exist
    required_splits = ['train', 'validation', 'test']
    for split in required_splits:
        if split not in dataset:
            validation_results["passed"] = False
            validation_results["checks"][f"{split}_exists"] = False
            raise ValueError(f"Missing required split: {split}")
        validation_results["checks"][f"{split}_exists"] = True
    
    # Check minimum sample counts
    min_samples = {'train': 1000, 'validation': 100, 'test': 100}
    for split, min_count in min_samples.items():
        count = len(dataset[split])
        validation_results["checks"][f"{split}_count"] = count
        if count < min_count:
            validation_results["warnings"].append(
                f"{split} has only {count} samples (expected >{min_count})"
            )
    
    # Check required columns
    required_columns = ['article', 'highlights']
    for split in required_splits:
        cols = dataset[split].column_names
        for col in required_columns:
            if col not in cols:
                validation_results["passed"] = False
                raise ValueError(f"Missing column '{col}' in {split} split")
        validation_results["checks"][f"{split}_columns"] = True
    
    # Check for null values
    for split in required_splits:
        sample = dataset[split][0]
        for col in required_columns:
            if sample[col] is None or sample[col] == "":
                validation_results["warnings"].append(
                    f"Found empty values in {col} column of {split} split"
                )
    
    # Check text length statistics
    for split in ['test']:
        articles = dataset[split]['article'][:100]
        lengths = [len(article.split()) for article in articles]
        
        avg_length = np.mean(lengths)
        validation_results["checks"][f"{split}_avg_article_length"] = avg_length
        
        if avg_length < 50:
            validation_results["warnings"].append(
                f"{split} articles are very short (avg: {avg_length:.0f} words)"
            )
    
    return validation_results


def calculate_model_size(model) -> Dict[str, float]:
    """
    Calculate model size and parameter count.
    
    Args:
        model: Transformer model
        
    Returns:
        Dictionary with model size metrics
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB (assuming float32)
    size_mb = (param_count * 4) / (1024 ** 2)
    
    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb
    }


def test_model_inference(
    summarizer,
    test_text: str,
    params: Dict[str, Any]
) -> Tuple[str, float]:
    """
    Test model inference and measure latency.
    
    Args:
        summarizer: Pipeline object
        test_text: Input text to summarize
        params: Model parameters
        
    Returns:
        Tuple of (summary, inference_time)
    """
    start_time = time.time()
    
    summary = summarizer(
        test_text,
        max_length=params['max_length'],
        min_length=params['min_length'],
        do_sample=params['do_sample']
    )[0]['summary_text']
    
    inference_time = time.time() - start_time
    
    return summary, inference_time


def train_model():
    """
    Load pretrained BART model with comprehensive quality assurance.
    
    This function implements:
    - CO2 emissions tracking with CodeCarbon
    - Data validation before training
    - Model testing and validation
    - Energy efficiency metrics
    - Comprehensive MLflow logging
    """
    # Initialize CO2 emissions tracker
    tracker = EmissionsTracker(
        project_name="bart_summarization",
        output_dir="reports/emissions",
        log_level="warning"
    )
    tracker.start()
    
    # Set MLflow tracking URI from environment
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Define parameters
    params = {
        "model_name": "facebook/bart-large-cnn",
        "task": "summarization",
        "max_length": 130,
        "min_length": 30,
        "do_sample": False,
        "test_samples": 5
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="bart_summarization_qa"):
        print("=" * 60)
        print("BART Summarization Pipeline - Quality Assurance Enabled")
        print("=" * 60)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Load dataset and validate
        print("\n[1/6] Loading and validating dataset...")
        data_path = Path("data/raw")
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Run 'dvc pull' first."
            )
        
        dataset = load_from_disk(str(data_path))
        
        # Validate dataset
        print("  → Running data validation checks...")
        validation_results = validate_dataset(dataset)
        
        if validation_results["passed"]:
            print("  ✓ Data validation passed")
        else:
            raise ValueError("Data validation failed")
        
        if validation_results["warnings"]:
            print("  ⚠ Warnings:")
            for warning in validation_results["warnings"]:
                print(f"    - {warning}")
        
        # Log validation results
        mlflow.log_dict(validation_results, "data_validation_results.json")
        mlflow.log_metrics({
            "train_samples": validation_results["checks"]["train_count"],
            "validation_samples": validation_results["checks"]["validation_count"],
            "test_samples": validation_results["checks"]["test_count"]
        })
        
        # Load model
        print(f"\n[2/6] Loading model: {params['model_name']}")
        start_load = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(params['model_name'])
        
        load_time = time.time() - start_load
        print(f"  ✓ Model loaded in {load_time:.2f}s")
        
        # Calculate and log model metrics
        print("\n[3/6] Analyzing model characteristics...")
        model_metrics = calculate_model_size(model)
        print(f"  → Parameters: {model_metrics['total_parameters']:,}")
        print(f"  → Model size: {model_metrics['model_size_mb']:.2f} MB")
        
        mlflow.log_metrics({
            "model_parameters": model_metrics['total_parameters'],
            "model_size_mb": model_metrics['model_size_mb'],
            "model_load_time_seconds": load_time
        })
        
        # Create pipeline
        summarizer = pipeline(
            params['task'],
            model=model,
            tokenizer=tokenizer
        )
        
        # Test model inference
        print("\n[4/6] Testing model inference...")
        test_samples = dataset['test'].select(
            range(min(params['test_samples'], len(dataset['test'])))
        )
        
        summaries = []
        inference_times = []
        
        for i, sample in enumerate(test_samples):
            article = sample['article']
            
            # Truncate if too long
            if len(article.split()) > 500:
                article = ' '.join(article.split()[:500])
            
            summary, inf_time = test_model_inference(summarizer, article, params)
            summaries.append(summary)
            inference_times.append(inf_time)
            
            print(f"  → Sample {i+1}/{params['test_samples']}: "
                  f"{len(sample['article'].split())} words → "
                  f"{len(summary.split())} words ({inf_time:.2f}s)")
        
        # Log inference metrics
        avg_inference_time = np.mean(inference_times)
        mlflow.log_metrics({
            "avg_inference_time_seconds": avg_inference_time,
            "max_inference_time_seconds": max(inference_times),
            "min_inference_time_seconds": min(inference_times)
        })
        
        print(f"  ✓ Average inference time: {avg_inference_time:.2f}s")
        
        # Save sample summaries
        summaries_file = "sample_summaries.txt"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            f.write("BART Summarization - Sample Outputs\n")
            f.write("=" * 60 + "\n\n")
            for i, (sample, summary) in enumerate(zip(test_samples, summaries)):
                f.write(f"Sample {i+1}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Article ({len(sample['article'].split())} words):\n")
                f.write(f"{sample['article'][:300]}...\n\n")
                f.write(f"Reference Summary:\n{sample['highlights']}\n\n")
                f.write(f"Generated Summary ({len(summary.split())} words):\n")
                f.write(f"{summary}\n\n")
                f.write("=" * 60 + "\n\n")
        
        mlflow.log_artifact(summaries_file)
        os.remove(summaries_file)
        
        # Save model
        print("\n[5/6] Saving model...")
        model_path = "models/bart_summarizer"
        os.makedirs(model_path, exist_ok=True)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Log model to MLflow
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model",
            task=params['task']
        )
        
        mlflow.log_param("model_saved_path", model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        # Stop emissions tracking
        print("\n[6/6] Collecting emissions data...")
        emissions = tracker.stop()
        
        # Log CO2 emissions and energy metrics
        emissions_data = {
            "co2_emissions_kg": emissions,
            "co2_emissions_g": emissions * 1000,
            "energy_consumed_kwh": tracker.final_emissions_data.energy_consumed,
            "duration_seconds": tracker.final_emissions_data.duration
        }
        
        mlflow.log_metrics(emissions_data)
        
        # Log emissions report if exists
        emissions_file = "reports/emissions/emissions.csv"
        if os.path.exists(emissions_file):
            mlflow.log_artifact(emissions_file, "emissions_report")
        
        print(f"  → CO2 emissions: {emissions * 1000:.2f}g")
        print(f"  → Energy consumed: {tracker.final_emissions_data.energy_consumed:.6f} kWh")
        print(f"  → Duration: {tracker.final_emissions_data.duration:.2f}s")
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            "co2_per_sample_g": (emissions * 1000) / params['test_samples'],
            "energy_per_sample_wh": (tracker.final_emissions_data.energy_consumed * 1000) / params['test_samples'],
            "time_per_sample_s": tracker.final_emissions_data.duration / params['test_samples']
        }
        mlflow.log_metrics(efficiency_metrics)
        
        print("\n" + "=" * 60)
        print("✓ Pipeline completed successfully!")
        print("=" * 60)
        print(f"\nEfficiency Summary:")
        print(f"  • CO2 per sample: {efficiency_metrics['co2_per_sample_g']:.2f}g")
        print(f"  • Energy per sample: {efficiency_metrics['energy_per_sample_wh']:.2f}Wh")
        print(f"  • Time per sample: {efficiency_metrics['time_per_sample_s']:.2f}s")
        print(f"Check MLflow UI for detailed metrics and artifacts")
        print(f"Emissions report saved to: reports/emissions/")


if __name__ == "__main__":
    train_model()
