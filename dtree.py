import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to match your actual directory structure
MODEL_DIR = r'D:\drdoProject\modelsAsPickle'
TEST_DATA_PATH = r'D:\test_Without_label.csv' 
# If your test data actually has labels for validation, specify the column name here.
# If it's purely for prediction (no labels), set this to None.
TARGET_COLUMN = 'label' 

# The specific Decision Tree models to test
MODEL_NAMES = ['dtree1', 'dtree2', 'dtree3']

def load_model(model_name):
    """Loads a pickled model from the specified directory."""
    path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        return None
    
    print(f"Loading {model_name}...")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def run_tests():
    # 1. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        # Create dummy data for demonstration if file is missing
        print("Generating dummy data for demonstration purposes...")
        df = pd.DataFrame(np.random.rand(100, 20), columns=[f'col_{i}' for i in range(20)])
        # Add a dummy label column if we are in validation mode
        if TARGET_COLUMN:
            df[TARGET_COLUMN] = np.random.randint(0, 2, 100)
    else:
        print(f"Reading test data from {TEST_DATA_PATH}...")
        df = pd.read_csv(TEST_DATA_PATH)

    # Prepare X (features) and y (labels)
    if TARGET_COLUMN and TARGET_COLUMN in df.columns:
        X_test = df.drop(columns=[TARGET_COLUMN])
        y_test = df[TARGET_COLUMN]
        has_labels = True
    else:
        X_test = df
        y_test = None
        has_labels = False
        print("No labels found. Running in Prediction-only mode.")

    # 2. Iterate through Decision Tree Models
    results = {}
    
    for name in MODEL_NAMES:
        print(f"\n--- Testing Model: {name} ---")
        model = load_model(name)
        
        if model is None:
            continue
            
        try:
            # Make Predictions
            predictions = model.predict(X_test)
            
            # Store predictions in the results dictionary
            results[name] = predictions
            
            # If we have labels, calculate metrics
            if has_labels:
                acc = accuracy_score(y_test, predictions)
                print(f"Accuracy: {acc:.4f}")
                print("Classification Report:")
                print(classification_report(y_test, predictions))
            else:
                print(f"Predictions generated for {len(predictions)} samples.")
                print(f"Sample predictions: {predictions[:5]}")

        except Exception as e:
            print(f"Failed to test {name}. Error: {e}")

    # 3. Save Predictions (Optional)
    if not has_labels:
        output_file = 'dtree_predictions.csv'
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False)
        print(f"\nAll predictions saved to {output_file}")

if __name__ == "__main__":
    run_tests()