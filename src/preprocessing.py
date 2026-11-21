"""Module for data preprocessing and feature extraction."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(
    text_data: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 7
) -> tuple:
    """
    Split the dataset into training and testing sets.
    
    Args:
        text_data: Series containing the news text.
        labels: Series containing the labels.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        text_data,
        labels,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Training set size: {len(x_train)}")
    print(f"Testing set size: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def vectorize_text(
    x_train: pd.Series,
    x_test: pd.Series,
    stop_words: str = 'english',
    max_df: float = 0.7
) -> tuple:
    """
    Vectorize text data using TF-IDF.
    
    Args:
        x_train: Training text data.
        x_test: Testing text data.
        stop_words: Stop words to filter out.
        max_df: Maximum document frequency threshold.
        
    Returns:
        Tuple of (tfidf_train, tfidf_test, vectorizer).
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=max_df)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    print(f"TF-IDF vectorizer vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Training features shape: {tfidf_train.shape}")
    print(f"Testing features shape: {tfidf_test.shape}")
    
    return tfidf_train, tfidf_test, tfidf_vectorizer
