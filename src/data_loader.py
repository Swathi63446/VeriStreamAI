"""Module for loading and exploring the news dataset."""

import pandas as pd
from pathlib import Path


def load_data(data_path: str = './data/news.csv') -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the news dataset from a CSV file.
    
    Args:
        data_path: Path to the CSV file containing news data.
        
    Returns:
        Tuple containing the DataFrame and labels Series.
    """
    df = pd.read_csv(data_path)
    return df


def get_dataset_info(df: pd.DataFrame) -> None:
    """
    Print dataset information and shape.
    
    Args:
        df: The news dataset DataFrame.
    """
    print(f"Dataset Shape: {df.shape}")
    print(f"\nFirst 5 records:\n{df.head()}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nData info:\n{df.info()}")


def get_labels(df: pd.DataFrame) -> pd.Series:
    """
    Extract labels from the DataFrame.
    
    Args:
        df: The news dataset DataFrame.
        
    Returns:
        Series containing the labels.
    """
    labels = df['label']
    print(f"\nLabels preview:\n{labels.head()}")
    print(f"Label value counts:\n{labels.value_counts()}")
    return labels
