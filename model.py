"""Module for model training and prediction."""

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def train_model(
    tfidf_train,
    y_train,
    max_iter: int = 50
) -> PassiveAggressiveClassifier:
    """
    Train a PassiveAggressiveClassifier model.
    
    Args:
        tfidf_train: TF-IDF vectorized training features.
        y_train: Training labels.
        max_iter: Maximum number of iterations.
        
    Returns:
        Trained model.
    """
    pac = PassiveAggressiveClassifier(max_iter=max_iter)
    pac.fit(tfidf_train, y_train)
    print(f"Model trained with {max_iter} iterations")
    return pac


def predict(model: PassiveAggressiveClassifier, tfidf_test) -> np.ndarray:
    """
    Make predictions on test data.
    
    Args:
        model: Trained PassiveAggressiveClassifier model.
        tfidf_test: TF-IDF vectorized test features.
        
    Returns:
        Predictions array.
    """
    predictions = model.predict(tfidf_test)
    return predictions


def evaluate_model(y_test, y_pred) -> float:
    """
    Evaluate model accuracy and print confusion matrix.
    
    Args:
        y_test: True test labels.
        y_pred: Predicted labels.
        
    Returns:
        Accuracy score.
    """
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {round(accuracy * 100, 2)}%")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print("\nConfusion Matrix:")
    print("                Predicted FAKE  Predicted REAL")
    print(f"Actual FAKE     {cm[0][0]}              {cm[0][1]}")
    print(f"Actual REAL     {cm[1][0]}              {cm[1][1]}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    return accuracy
