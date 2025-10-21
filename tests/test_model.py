from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from evaluate import load as load_metric
from src.modeling.predict import predict_model

MODEL_PATH = "models/bart-large-cnn-finetuned"
DATA_PATH = "data/raw"


def test_model_files_exist():
    path = Path(MODEL_PATH)
    assert path.exists()
    assert (path / "config.json").exists()
    assert (path / "tokenizer_config.json").exists()


def test_data_splits_and_columns():
    dataset = load_from_disk(DATA_PATH)
    expected_columns = ["article", "highlights", "id"]
    for split in ["train", "validation", "test"]:
        assert split in dataset
        cols = dataset[split].column_names
        for col in expected_columns:
            assert col in cols
        sample = dataset[split][0]
        for col in expected_columns:
            assert sample[col] is not None


def test_rouge_scores():
    dataset = load_from_disk(DATA_PATH)
    rouge = load_metric("rouge")
    
    predictions, references = [], []
    for i in range(min(1, len(dataset["test"]))):
        sample = dataset["test"][i]
        input_text = sample["article"]
        reference = sample["highlights"]
        prediction = predict_model(input_text, MODEL_PATH)
        predictions.append(prediction)
        references.append(reference)
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    assert result["rouge1"] * 100 > 35.0
    assert result["rouge2"] * 100 > 15.0
    assert result["rougeL"] * 100 > 20.0


def test_summary_length():
    dataset = load_from_disk(DATA_PATH)
    
    lengths = []
    for i in range(min(1, len(dataset["test"]))):
        sample = dataset["test"][i]
        input_text = sample["article"]
        prediction = predict_model(input_text, MODEL_PATH)
        lengths.append(len(prediction.split()))
    
    avg_len = np.mean(lengths)
    assert 5 < avg_len < 150  # Adjust lower bound for very short articles


def test_text_edge_cases():
    short_text = "Climate scientists reported record temperatures."
    long_text = "Climate change affects temperatures. " * 200
    empty_texts = ["", "   "]
    
    # Short text
    summary_short = predict_model(short_text, MODEL_PATH)
    assert summary_short and isinstance(summary_short, str)
    
    # Long text
    summary_long = predict_model(long_text, MODEL_PATH)
    assert summary_long and len(summary_long) > 0
    
    # Empty text
    for t in empty_texts:
        assert predict_model(t, MODEL_PATH) == ""


def test_deterministic_prediction():
    text = "Scientists discover new climate patterns."
    summary1 = predict_model(text, MODEL_PATH)
    summary2 = predict_model(text, MODEL_PATH)
    assert summary1 == summary2


def test_predict_signature():
    import inspect
    sig = inspect.signature(predict_model)
    params = list(sig.parameters.keys())
    assert "text" in params
    assert "model_path" in params
