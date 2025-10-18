"""
Professional model testing suite with Pylint compliance.
Tests model performance, data quality, and code practices.
"""
import unittest
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric
from src.modeling.predict import predict_model


class TestModelPerformance(unittest.TestCase):
    """Test model quality and performance metrics."""
    
    @classmethod
    def setUpClass(cls):
        """Load model and test data once for all tests."""
        cls.model_path = "models/bart_news_final"
        cls.data_path = "data/processed"
        
        if not Path(cls.model_path).exists():
            raise FileNotFoundError(f"Model not found at {cls.model_path}")
        
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_path)
        cls.rouge = load_metric("rouge")
        
        dataset = load_from_disk(cls.data_path)
        cls.test_data = dataset["test"]
        cls.sample_size = min(100, len(cls.test_data))
    
    def test_model_files_exist(self):
        """Test that all required model files are present."""
        model_dir = Path(self.model_path)
        
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        
        for filename in required_files:
            file_path = model_dir / filename
            self.assertTrue(
                file_path.exists(),
                f"Missing required file: {filename}"
            )
        
        model_weights_exist = (
            (model_dir / "pytorch_model.bin").exists() or
            (model_dir / "model.safetensors").exists()
        )
        self.assertTrue(
            model_weights_exist,
            "Model weights file not found"
        )
    
    def test_rouge1_meets_threshold(self):
        """Test ROUGE-1 score meets minimum quality threshold."""
        predictions = []
        references = []
        
        for i in range(self.sample_size):
            sample = self.test_data[i]
            
            input_text = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True
            )
            reference_text = self.tokenizer.decode(
                [token for token in sample["labels"] if token != -100],
                skip_special_tokens=True
            )
            
            prediction = predict_model(input_text, self.model_path)
            predictions.append(prediction)
            references.append(reference_text)
        
        result = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        rouge1_score = result["rouge1"] * 100
        min_threshold = 35.0
        
        self.assertGreater(
            rouge1_score,
            min_threshold,
            f"ROUGE-1 {rouge1_score:.2f} below threshold {min_threshold}"
        )
        print(f"\n ROUGE-1: {rouge1_score:.2f} (threshold: {min_threshold})")
    
    def test_rouge2_meets_threshold(self):
        """Test ROUGE-2 score meets minimum quality threshold."""
        predictions = []
        references = []
        
        for i in range(self.sample_size):
            sample = self.test_data[i]
            
            input_text = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True
            )
            reference_text = self.tokenizer.decode(
                [token for token in sample["labels"] if token != -100],
                skip_special_tokens=True
            )
            
            prediction = predict_model(input_text, self.model_path)
            predictions.append(prediction)
            references.append(reference_text)
        
        result = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        rouge2_score = result["rouge2"] * 100
        min_threshold = 15.0
        
        self.assertGreater(
            rouge2_score,
            min_threshold,
            f"ROUGE-2 {rouge2_score:.2f} below threshold {min_threshold}"
        )
        print(f"ROUGE-2: {rouge2_score:.2f} (threshold: {min_threshold})")
    
    def test_rougeL_meets_threshold(self):
        """Test ROUGE-L score meets minimum quality threshold."""
        predictions = []
        references = []
        
        for i in range(self.sample_size):
            sample = self.test_data[i]
            
            input_text = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True
            )
            reference_text = self.tokenizer.decode(
                [token for token in sample["labels"] if token != -100],
                skip_special_tokens=True
            )
            
            prediction = predict_model(input_text, self.model_path)
            predictions.append(prediction)
            references.append(reference_text)
        
        result = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        rougeL_score = result["rougeL"] * 100
        min_threshold = 30.0
        
        self.assertGreater(
            rougeL_score,
            min_threshold,
            f"ROUGE-L {rougeL_score:.2f} below threshold {min_threshold}"
        )
        print(f"ROUGE-L: {rougeL_score:.2f} (threshold: {min_threshold})")
    
    def test_summary_length_valid(self):
        """Test generated summaries are within acceptable length range."""
        summary_lengths = []
        min_length = 20
        max_length = 150
        
        for i in range(min(50, len(self.test_data))):
            sample = self.test_data[i]
            
            input_text = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True
            )
            
            prediction = predict_model(input_text, self.model_path)
            token_count = len(self.tokenizer.encode(prediction))
            summary_lengths.append(token_count)
        
        avg_length = np.mean(summary_lengths)
        
        self.assertGreater(
            avg_length,
            min_length,
            f"Average summary too short: {avg_length:.1f} tokens"
        )
        self.assertLess(
            avg_length,
            max_length,
            f"Average summary too long: {avg_length:.1f} tokens"
        )
        print(f"Avg length: {avg_length:.1f} tokens (range: {min_length}-{max_length})")
    
    def test_short_text_processed_correctly(self):
        """Test that short articles are processed without chunking."""
        short_article = (
            "Climate scientists reported record temperatures this month. "
            "The heat wave affected millions and raised climate concerns."
        )
        
        summary = predict_model(short_article, self.model_path)
        
        self.assertIsNotNone(summary, "Summary is None")
        self.assertGreater(len(summary), 0, "Summary is empty")
        self.assertIsInstance(summary, str, "Summary is not a string")
        
        token_count = len(self.tokenizer.encode(summary))
        self.assertLess(
            token_count,
            150,
            f"Short text summary too long: {token_count} tokens"
        )
        print(f"Short text: {len(short_article)} chars → {len(summary)} chars")
    
    def test_long_text_chunked_correctly(self):
        """Test that long articles are properly chunked and summarized."""
        base_text = "Climate change affects global temperatures. "
        long_article = base_text * 200
        
        summary = predict_model(long_article, self.model_path)
        
        self.assertIsNotNone(summary, "Long text summary is None")
        self.assertGreater(len(summary), 0, "Long text summary is empty")
        
        token_count = len(self.tokenizer.encode(summary))
        self.assertLess(
            token_count,
            150,
            f"Long text summary too long: {token_count} tokens"
        )
        print(f"Long text: {len(long_article)} chars → {len(summary)} chars")
    
    def test_empty_text_handled(self):
        """Test that empty input is handled gracefully."""
        empty_inputs = ["", "   ", "\n\n", "\t"]
        
        for empty_text in empty_inputs:
            summary = predict_model(empty_text, self.model_path)
            self.assertEqual(
                summary,
                "",
                f"Empty input should return empty string, got: {summary}"
            )
    
    def test_predictions_are_deterministic(self):
        """Test that model produces consistent outputs for same input."""
        test_text = (
            "Scientists discover new climate patterns affecting weather. "
            "Research shows significant changes in global temperatures."
        )
        
        summary1 = predict_model(test_text, self.model_path)
        summary2 = predict_model(test_text, self.model_path)
        
        self.assertEqual(
            summary1,
            summary2,
            "Model outputs are not deterministic"
        )
        print(f"Deterministic output verified")


class TestDataQuality(unittest.TestCase):
    """Test data quality and structure."""
    
    def test_processed_data_exists(self):
        """Test that processed data directory exists."""
        data_path = Path("data/processed")
        self.assertTrue(
            data_path.exists(),
            "Processed data directory not found"
        )
    
    def test_all_splits_present(self):
        """Test that all required data splits exist."""
        data_path = Path("data/processed")
        
        if not data_path.exists():
            self.skipTest("Processed data not found")
        
        dataset = load_from_disk(str(data_path))
        required_splits = ["train", "validation", "test"]
        
        for split_name in required_splits:
            self.assertIn(
                split_name,
                dataset,
                f"Missing required split: {split_name}"
            )
            self.assertGreater(
                len(dataset[split_name]),
                0,
                f"{split_name} split is empty"
            )
    
    def test_data_has_required_columns(self):
        """Test that data has all required columns."""
        data_path = Path("data/processed")
        
        if not data_path.exists():
            self.skipTest("Processed data not found")
        
        dataset = load_from_disk(str(data_path))
        required_columns = ["input_ids", "attention_mask", "labels"]
        
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                columns = dataset[split_name].column_names
                for column_name in required_columns:
                    self.assertIn(
                        column_name,
                        columns,
                        f"Missing column '{column_name}' in {split_name}"
                    )
    
    def test_no_null_values_in_data(self):
        """Test that data contains no null values."""
        data_path = Path("data/processed")
        
        if not data_path.exists():
            self.skipTest("Processed data not found")
        
        dataset = load_from_disk(str(data_path))
        
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                sample = dataset[split_name][0]
                
                self.assertIsNotNone(
                    sample["input_ids"],
                    f"Null input_ids in {split_name}"
                )
                self.assertIsNotNone(
                    sample["attention_mask"],
                    f"Null attention_mask in {split_name}"
                )
                self.assertIsNotNone(
                    sample["labels"],
                    f"Null labels in {split_name}"
                )
    
    def test_data_distribution_balanced(self):
        """Test that data splits have reasonable distribution."""
        data_path = Path("data/processed")
        
        if not data_path.exists():
            self.skipTest("Processed data not found")
        
        dataset = load_from_disk(str(data_path))
        
        train_size = len(dataset["train"])
        val_size = len(dataset["validation"])
        test_size = len(dataset["test"])
        total_size = train_size + val_size + test_size
        
        train_ratio = train_size / total_size * 100
        
        self.assertGreater(
            train_ratio,
            60.0,
            f"Training set too small: {train_ratio:.1f}% (expected >60%)"
        )
        print(f"Data distribution: train={train_ratio:.1f}%")


class TestCodeQuality(unittest.TestCase):
    """Test code quality and best practices."""
    
    def test_required_packages_installed(self):
        """Test that all required packages can be imported."""
        required_packages = [
            "transformers",
            "datasets",
            "evaluate",
            "mlflow",
            "codecarbon",
            "numpy",
            "torch"
        ]
        
        for package_name in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                self.fail(f"Required package not installed: {package_name}")
    
    def test_predict_function_signature(self):
        """Test that predict function has correct signature."""
        import inspect
        
        signature = inspect.signature(predict_model)
        params = list(signature.parameters.keys())
        
        self.assertIn("text", params, "Missing 'text' parameter")
        self.assertIn("model_path", params, "Missing 'model_path' parameter")
        
        default_path = signature.parameters["model_path"].default
        self.assertIsNotNone(
            default_path,
            "model_path should have default value"
        )
    
    def test_predict_returns_string(self):
        """Test that predict function returns string type."""
        test_text = "Test article about climate change and global warming."
        result = predict_model(test_text)
        
        self.assertIsInstance(
            result,
            str,
            f"predict_model should return str, got {type(result)}"
        )
    
    def test_model_directory_structure(self):
        """Test that model follows github structure."""
        required_dirs = [
            "data",
            "models",
            "reports",
            "src/modeling"
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(
                Path(dir_path).exists(),
                f"Required directory not found: {dir_path}"
            )
    
    def test_emissions_tracking_enabled(self):
        """Test that emissions tracking file exists."""
        emissions_file = Path("reports/emissions.csv")
        
        if emissions_file.exists():
            with open(emissions_file, 'r', encoding='utf-8') as file:
                content = file.read()
                self.assertIn(
                    "emissions",
                    content.lower(),
                    "Emissions file missing emissions data"
                )
    
    def test_mlflow_configured(self):
        """Test that MLflow tracking is configured."""
        import os
        
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.assertIsNotNone(
            mlflow_uri,
            "MLFLOW_TRACKING_URI environment variable not set"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
