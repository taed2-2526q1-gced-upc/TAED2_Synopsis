from pathlib import Path
from datasets import load_dataset
from evaluate import load as load_metric
from src.modeling.predict import predict_model

MLFLOW_ARTIFACT_URI = (
    "mlflow-artifacts:/9b4993739fa24d1b80a3e605434b618d/"
    "98108604288f454f89806e859eaf71f8/artifacts/bart-large-cnn-finetuned"
)

MLFLOW_ARTIFACT_URI = "models/bart-large-cnn-finetuned"


def test_model_files_exist():
    path = Path("models/bart-large-cnn-finetuned")
    assert path.exists()
    assert (path / "config.json").exists()
    assert (path / "tokenizer_config.json").exists()


def test_data_splits_and_columns():
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/raw/train.parquet",
            "validation": "data/raw/validation.parquet",
            "test": "data/raw/test.parquet",
        }
    )
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
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/raw/train.parquet",
            "validation": "data/raw/validation.parquet",
            "test": "data/raw/test.parquet",
        }
    )
    rouge = load_metric("rouge")
    
    predictions, references = [], []
    for i in range(min(1, len(dataset["test"]))):
        sample = dataset["test"][i]
        input_text = sample["article"]
        reference = sample["highlights"]
        prediction = predict_model(input_text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
        predictions.append(prediction)
        references.append(reference)
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    assert result["rouge1"] * 100 > 35.0
    assert result["rouge2"] * 100 > 15.0
    assert result["rougeL"] * 100 > 20.0


def test_summary_length():
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/raw/train.parquet",
            "validation": "data/raw/validation.parquet",
            "test": "data/raw/test.parquet",
        }
    )
    
    lengths = []
    for i in range(min(1, len(dataset["test"]))):
        sample = dataset["test"][i]
        input_text = sample["article"]
        prediction = predict_model(input_text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
        lengths.append(len(prediction.split()))
        
        input_len = len(input_text.split())
        min_len = min(5, int(input_len * 0.1))   
        max_len = max(150, int(input_len * 0.5)) 
        pred_len = len(prediction.split())
        assert min_len < pred_len < max_len


def test_short_text():
    short_text = "Climate scientists reported record temperatures."
    summary_short = predict_model(short_text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
    assert summary_short and isinstance(summary_short, str)


def test_long_text():
    long_text = "Climate change affects temperatures. " * 200
    summary_long = predict_model(long_text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
    assert summary_long and len(summary_long) > 0


def test_empty_text():
    empty_texts = ["", "   "]
    for t in empty_texts:
        assert predict_model(t, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI) == ""


def test_deterministic_prediction():
    text = "Scientists discover new climate patterns."
    summary1 = predict_model(text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
    summary2 = predict_model(text, mlflow_artifact_uri=MLFLOW_ARTIFACT_URI)
    assert summary1 == summary2


def test_predict_signature():
    import inspect
    sig = inspect.signature(predict_model)
    params = list(sig.parameters.keys())
    assert "text" in params
    assert "mlflow_artifact_uri" in params
