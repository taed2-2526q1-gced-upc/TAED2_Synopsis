"""
Test model performance and quality.
"""
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric
from src.modeling.predict import predict_model


def test_model_files_exist():
    """Test model files are present."""
    model_path = Path("models/bart_news_final")
    assert model_path.exists()
    assert (model_path / "config.json").exists()
    assert (model_path / "tokenizer_config.json").exists()


def test_rouge1_threshold():
    """Test ROUGE-1 meets minimum threshold."""
    model_path = "models/bart_news_final"
    data_path = "data/processed"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_from_disk(data_path)
    rouge = load_metric("rouge")
    
    predictions = []
    references = []
    
    for i in range(min(100, len(dataset["test"]))):
        sample = dataset["test"][i]
        
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(
            [t for t in sample["labels"] if t != -100],
            skip_special_tokens=True
        )
        
        prediction = predict_model(input_text, model_path)
        predictions.append(prediction)
        references.append(reference)
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge1 = result["rouge1"] * 100
    
    assert rouge1 > 35.0


def test_rouge2_threshold():
    """Test ROUGE-2 meets minimum threshold."""
    model_path = "models/bart_news_final"
    data_path = "data/processed"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_from_disk(data_path)
    rouge = load_metric("rouge")
    
    predictions = []
    references = []
    
    for i in range(min(100, len(dataset["test"]))):
        sample = dataset["test"][i]
        
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(
            [t for t in sample["labels"] if t != -100],
            skip_special_tokens=True
        )
        
        prediction = predict_model(input_text, model_path)
        predictions.append(prediction)
        references.append(reference)
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge2 = result["rouge2"] * 100
    
    assert rouge2 > 15.0


def test_rougeL_threshold():
    """Test ROUGE-L meets minimum threshold."""
    model_path = "models/bart_news_final"
    data_path = "data/processed"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_from_disk(data_path)
    rouge = load_metric("rouge")
    
    predictions = []
    references = []
    
    for i in range(min(100, len(dataset["test"]))):
        sample = dataset["test"][i]
        
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(
            [t for t in sample["labels"] if t != -100],
            skip_special_tokens=True
        )
        
        prediction = predict_model(input_text, model_path)
        predictions.append(prediction)
        references.append(reference)
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rougeL = result["rougeL"] * 100
    
    assert rougeL > 30.0


def test_summary_length():
    """Test summaries have valid length."""
    model_path = "models/bart_news_final"
    data_path = "data/processed"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_from_disk(data_path)
    
    lengths = []
    for i in range(min(50, len(dataset["test"]))):
        sample = dataset["test"][i]
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        prediction = predict_model(input_text, model_path)
        token_count = len(tokenizer.encode(prediction))
        lengths.append(token_count)
    
    avg_length = np.mean(lengths)
    assert avg_length > 20
    assert avg_length < 150


def test_short_text():
    """Test short text processing."""
    text = "Climate scientists reported record temperatures."
    summary = predict_model(text)
    
    assert summary is not None
    assert len(summary) > 0
    assert isinstance(summary, str)


def test_long_text():
    """Test long text chunking."""
    text = "Climate change affects temperatures. " * 200
    summary = predict_model(text)
    
    assert summary is not None
    assert len(summary) > 0


def test_empty_text():
    """Test empty text handling."""
    assert predict_model("") == ""
    assert predict_model("   ") == ""


def test_deterministic():
    """Test predictions are consistent."""
    text = "Scientists discover new climate patterns."
    summary1 = predict_model(text)
    summary2 = predict_model(text)
    
    assert summary1 == summary2


def test_data_splits_exist():
    """Test all data splits present."""
    data_path = Path("data/processed")
    assert data_path.exists()
    
    dataset = load_from_disk(str(data_path))
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset


def test_data_columns():
    """Test data has required columns."""
    dataset = load_from_disk("data/processed")
    
    for split in ["train", "validation", "test"]:
        columns = dataset[split].column_names
        assert "input_ids" in columns
        assert "attention_mask" in columns
        assert "labels" in columns


def test_no_nulls():
    """Test data has no null values."""
    dataset = load_from_disk("data/processed")
    
    for split in ["train", "validation", "test"]:
        sample = dataset[split][0]
        assert sample["input_ids"] is not None
        assert sample["attention_mask"] is not None
        assert sample["labels"] is not None


def test_predict_signature():
    """Test predict function signature."""
    import inspect
    
    sig = inspect.signature(predict_model)
    params = list(sig.parameters.keys())
    
    assert "text" in params
    assert "model_path" in params
