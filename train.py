"""
Training script for Fake News Detection Model

This script loads the data, preprocesses it, trains the model, and saves it.
"""

from src.data_loader import load_data, get_dataset_info, get_labels
from src.preprocessing import split_data, vectorize_text
from src.model import train_model, predict, evaluate_model, save_model


def main():
    """Train and save the fake news detection model."""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load the data
    print("\n[STEP 1] Loading Data...")
    print("-" * 60)
    df = load_data('./data/news.csv')
    get_dataset_info(df)
    
    # Step 2: Extract labels
    print("\n[STEP 2] Extracting Labels...")
    print("-" * 60)
    labels = get_labels(df)
    
    # Step 3: Split the dataset
    print("\n[STEP 3] Splitting Data into Train/Test Sets...")
    print("-" * 60)
    x_train, x_test, y_train, y_test = split_data(
        df['text'],
        labels,
        test_size=0.2,
        random_state=7
    )
    
    # Step 4: Vectorize text data
    print("\n[STEP 4] Vectorizing Text Data with TF-IDF...")
    print("-" * 60)
    tfidf_train, tfidf_test, vectorizer = vectorize_text(
        x_train,
        x_test,
        stop_words='english',
        max_df=0.7
    )
    
    # Step 5: Train the model
    print("\n[STEP 5] Training PassiveAggressiveClassifier...")
    print("-" * 60)
    model = train_model(tfidf_train, y_train, max_iter=50)
    
    # Step 6: Make predictions
    print("\n[STEP 6] Making Predictions...")
    print("-" * 60)
    y_pred = predict(model, tfidf_test)
    
    # Step 7: Evaluate the model
    print("\n[STEP 7] Evaluating Model Performance...")
    print("-" * 60)
    accuracy = evaluate_model(y_test, y_pred)
    
    # Step 8: Save the model
    print("\n[STEP 8] Saving Model and Vectorizer...")
    print("-" * 60)
    save_model(model, vectorizer, './models')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Accuracy: {round(accuracy * 100, 2)}%")
    print("=" * 60)
    
    return model, vectorizer, accuracy


if __name__ == "__main__":
    main()
