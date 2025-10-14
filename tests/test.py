"""
Data quality testing using Great Expectations and custom validators.

Tests include:
- Schema validation
- Data quality checks
- Statistical validation
- Content validation
"""
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, List
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from datasets import load_from_disk
import numpy as np


class TestDatasetStructure:
    """Test suite for dataset structure validation."""
    
    @pytest.fixture(scope="class")
    def dataset(self):
        """Load dataset for testing."""
        data_path = Path("data/raw")
        if not data_path.exists():
            pytest.skip("Dataset not found")
        
        return load_from_disk(str(data_path))
    
    def test_required_splits_exist(self, dataset):
        """Test that all required splits are present."""
        required_splits = ['train', 'validation', 'test']
        
        for split in required_splits:
            assert split in dataset, f"Missing split: {split}"
    
    def test_required_columns_exist(self, dataset):
        """Test that required columns are present."""
        required_columns = ['article', 'highlights']
        
        for split in ['train', 'validation', 'test']:
            columns = dataset[split].column_names
            for col in required_columns:
                assert col in columns, \
                    f"Missing column '{col}' in {split} split"
    
    def test_dataset_size_requirements(self, dataset):
        """Test minimum dataset sizes."""
        min_sizes = {
            'train': 1000,
            'validation': 100,
            'test': 100
        }
        
        for split, min_size in min_sizes.items():
            actual_size = len(dataset[split])
            assert actual_size >= min_size, \
                f"{split} split too small: {actual_size} < {min_size}"
    
    def test_no_duplicate_indices(self, dataset):
        """Test that there are no duplicate samples."""
        for split in ['train', 'validation', 'test']:
            size = len(dataset[split])
            # Check a sample if dataset is large
            if size > 1000:
                sample_size = 1000
                indices = np.random.choice(size, sample_size, replace=False)
                articles = [dataset[split][int(i)]['article'] for i in indices]
            else:
                articles = dataset[split]['article']
            
            unique_articles = len(set(articles))
            total_articles = len(articles)
            
            # Allow up to 5% duplicates
            assert unique_articles >= total_articles * 0.95, \
                f"{split} has too many duplicates"


class TestDataQuality:
    """Test suite for data quality checks."""
    
    @pytest.fixture(scope="class")
    def dataset(self):
        """Load dataset for testing."""
        data_path = Path("data/raw")
        if not data_path.exists():
            pytest.skip("Dataset not found")
        
        return load_from_disk(str(data_path))
    
    def test_no_null_values(self, dataset):
        """Test that there are no null values in required fields."""
        for split in ['train', 'validation', 'test']:
            sample_size = min(100, len(dataset[split]))
            samples = dataset[split].select(range(sample_size))
            
            for i, sample in enumerate(samples):
                assert sample['article'] is not None, \
                    f"Null article in {split}[{i}]"
                assert sample['highlights'] is not None, \
                    f"Null highlights in {split}[{i}]"
                assert sample['article'] != "", \
                    f"Empty article in {split}[{i}]"
                assert sample['highlights'] != "", \
                    f"Empty highlights in {split}[{i}]"
    
    def test_text_length_distribution(self, dataset):
        """Test that text lengths are within reasonable bounds."""
        for split in ['train', 'test']:
            sample_size = min(100, len(dataset[split]))
            samples = dataset[split].select(range(sample_size))
            
            article_lengths = [len(s['article'].split()) for s in samples]
            summary_lengths = [len(s['highlights'].split()) for s in samples]
            
            # Articles should have reasonable length
            avg_article_len = np.mean(article_lengths)
            assert avg_article_len > 50, \
                f"{split} articles too short: avg {avg_article_len:.0f} words"
            assert avg_article_len < 5000, \
                f"{split} articles too long: avg {avg_article_len:.0f} words"
            
            # Summaries should be shorter than articles
            avg_summary_len = np.mean(summary_lengths)
            assert avg_summary_len < avg_article_len, \
                "Summaries longer than articles"
            
            print(f"\n{split} stats:")
            print(f"  Avg article length: {avg_article_len:.0f} words")
            print(f"  Avg summary length: {avg_summary_len:.0f} words")
    
    def test_text_encoding(self, dataset):
        """Test that text is properly encoded."""
        sample = dataset['train'][0]
        
        # Should be string type
        assert isinstance(sample['article'], str)
        assert isinstance(sample['highlights'], str)
        
        # Should be valid UTF-8
        try:
            sample['article'].encode('utf-8')
            sample['highlights'].encode('utf-8')
        except UnicodeEncodeError:
            pytest.fail("Text contains invalid UTF-8 characters")
    
    def test_no_excessive_whitespace(self, dataset):
        """Test that text doesn't have excessive whitespace."""
        sample_size = min(50, len(dataset['train']))
        samples = dataset['train'].select(range(sample_size))
        
        for i, sample in enumerate(samples):
            article = sample['article']
            highlights = sample['highlights']
            
            assert '   ' not in article, \
                f"Excessive whitespace in train[{i}] article"
            assert '   ' not in highlights, \
                f"Excessive whitespace in train[{i}] highlights"
    
    def test_summary_quality(self, dataset):
        """Test basic summary quality metrics."""
        sample_size = min(50, len(dataset['test']))
        samples = dataset['test'].select(range(sample_size))
        
        for i, sample in enumerate(samples):
            summary = sample['highlights']
            article = sample['article']
            
            article_words = set(article.lower().split()[:50])
            summary_words = set(summary.lower().split())
            
            if len(summary_words) > 0:
                overlap = len(article_words & summary_words) / len(summary_words)
                assert overlap < 0.95, \
                    f"Summary appears to be copied from article in test[{i}]"


class TestGreatExpectations:
    """Test suite using Great Expectations framework."""
    
    @pytest.fixture(scope="class")
    def gx_context(self):
        """Initialize Great Expectations context."""
        context = gx.get_context()
        return context
    
    @pytest.fixture(scope="class")
    def train_df(self):
        """Load training data as DataFrame."""
        data_path = Path("data/raw")
        if not data_path.exists():
            pytest.skip("Dataset not found")
        
        dataset = load_from_disk(str(data_path))
        
        # Convert to DataFrame (sample for efficiency)
        sample_size = min(1000, len(dataset['train']))
        samples = dataset['train'].select(range(sample_size))
        
        df = pd.DataFrame({
            'article': samples['article'],
            'highlights': samples['highlights']
        })
        
        df['article_word_count'] = df['article'].str.split().str.len()
        df['summary_word_count'] = df['highlights'].str.split().str.len()
        df['compression_ratio'] = df['article_word_count'] / df['summary_word_count']
        
        return df
    
    def test_ge_column_existence(self, train_df):
        """Test column existence with Great Expectations."""
        required_columns = ['article', 'highlights']
        
        for col in required_columns:
            assert col in train_df.columns, f"Missing column: {col}"
    
    def test_ge_no_null_values(self, train_df):
        """Test for null values using GE."""
        null_counts = train_df[['article', 'highlights']].isnull().sum()
        
        assert null_counts['article'] == 0, \
            f"Found {null_counts['article']} null articles"
        assert null_counts['highlights'] == 0, \
            f"Found {null_counts['highlights']} null summaries"
    
    def test_ge_word_count_range(self, train_df):
        """Test word count statistics."""
        # Articles should have reasonable word counts
        assert train_df['article_word_count'].min() > 10, \
            "Some articles are too short"
        assert train_df['article_word_count'].max() < 10000, \
            "Some articles are too long"
        
        # Summaries should be shorter
        assert train_df['summary_word_count'].min() > 5, \
            "Some summaries are too short"
        assert train_df['summary_word_count'].max() < 500, \
            "Some summaries are too long"
    
    def test_ge_compression_ratio(self, train_df):
        """Test compression ratio is reasonable."""
        median_ratio = train_df['compression_ratio'].median()
        
        # Typical news summaries compress 10-20x
        assert 3 < median_ratio < 50, \
            f"Unusual compression ratio: {median_ratio:.2f}"
        
        print(f"\nCompression ratio: {median_ratio:.2f}x")
    
    def test_ge_text_patterns(self, train_df):
        """Test for common text quality issues."""
        url_pattern = r'http[s]?://'
        summaries_with_urls = train_df['highlights'].str.contains(
            url_pattern, 
            regex=True, 
            na=False
        ).sum()
        
        url_ratio = summaries_with_urls / len(train_df)
        assert url_ratio < 0.05, \
            f"Too many summaries contain URLs: {url_ratio:.2%}"
    
    def test_ge_statistical_distribution(self, train_df):
        """Test statistical properties of the data."""
        q1 = train_df['article_word_count'].quantile(0.25)
        q3 = train_df['article_word_count'].quantile(0.75)
        iqr = q3 - q1
        
        outliers = train_df[
            (train_df['article_word_count'] < q1 - 3*iqr) |
            (train_df['article_word_count'] > q3 + 3*iqr)
        ]
        
        outlier_ratio = len(outliers) / len(train_df)
        assert outlier_ratio < 0.05, \
            f"Too many outliers in article lengths: {outlier_ratio:.2%}"


class TestDataConsistency:
    """Test suite for data consistency across splits."""
    
    @pytest.fixture(scope="class")
    def dataset(self):
        """Load dataset for testing."""
        data_path = Path("data/raw")
        if not data_path.exists():
            pytest.skip("Dataset not found")
        
        return load_from_disk(str(data_path))
    
    def test_split_distribution_similarity(self, dataset):
        """Test that splits have similar distributions."""
        splits = ['train', 'validation', 'test']
        stats = {}
        
        for split in splits:
            sample_size = min(100, len(dataset[split]))
            samples = dataset[split].select(range(sample_size))
            
            article_lengths = [len(s['article'].split()) for s in samples]
            stats[split] = {
                'mean': np.mean(article_lengths),
                'std': np.std(article_lengths)
            }
        
        train_mean = stats['train']['mean']
        for split in ['validation', 'test']:
            split_mean = stats[split]['mean']
            ratio = abs(split_mean - train_mean) / train_mean
            
            assert ratio < 0.3, \
                f"{split} distribution differs significantly from train: {ratio:.2%}"
    
    def test_no_data_leakage(self, dataset):
        """Test that there's no overlap between splits."""
        # Sample articles from each split
        train_sample = dataset['train'].select(range(min(100, len(dataset['train']))))
        test_sample = dataset['test'].select(range(min(100, len(dataset['test']))))
        
        train_articles = set(train_sample['article'])
        test_articles = set(test_sample['article'])
        
        overlap = train_articles & test_articles
        
        assert len(overlap) == 0, \
            f"Found {len(overlap)} duplicate articles between train and test"


def generate_data_quality_report(dataset_path: str = "data/raw"):
    """
    Generate comprehensive data quality report.
    
    Args:
        dataset_path: Path to the dataset
    """
    from datetime import datetime
    
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_path}")
    print()
    
    try:
        dataset = load_from_disk(dataset_path)
        
        print("Dataset Structure:")
        print("-" * 70)
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                print(f"  {split:12s}: {len(dataset[split]):,} samples")
                print(f"                Columns: {dataset[split].column_names}")
        print()
        
        print("Statistical Summary:")
        print("-" * 70)
        for split in ['train', 'test']:
            if split in dataset:
                sample_size = min(500, len(dataset[split]))
                samples = dataset[split].select(range(sample_size))
                
                article_lengths = [len(s['article'].split()) for s in samples]
                summary_lengths = [len(s['highlights'].split()) for s in samples]
                
                print(f"\n  {split.upper()} Split:")
                print(f"    Article length (words):")
                print(f"      Mean: {np.mean(article_lengths):.0f}")
                print(f"      Median: {np.median(article_lengths):.0f}")
                print(f"      Std: {np.std(article_lengths):.0f}")
                print(f"      Min: {np.min(article_lengths)}")
                print(f"      Max: {np.max(article_lengths)}")
                print(f"    Summary length (words):")
                print(f"      Mean: {np.mean(summary_lengths):.0f}")
                print(f"      Median: {np.median(summary_lengths):.0f}")
                print(f"    Compression ratio: {np.mean(article_lengths)/np.mean(summary_lengths):.2f}x")
        
        print("\n" + "=" * 70)
        print("✓ Data quality report generated successfully")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Generate report
    print("\n")
    generate_data_quality_report()
