"""
Test script to verify src/prediction.py on a real processed ECG heartbeat sample.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.prediction import predict_single_beat


def test_prediction():
    print("=" * 60)
    print("        TESTING PREDICTION MODULE (src/prediction.py)")
    print("=" * 60)

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "processed_beats.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Processed dataset not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    sample_row = df.iloc[0]
    
    feature_cols = [c for c in df.columns if c.startswith('s_')]
    signal_vector = sample_row[feature_cols].values.tolist()
    true_label = sample_row['aami_class']
    record_id = sample_row['record_id']

    print(f"\n[SAMPLE TEST] Record ID: {record_id} | True AAMI Class: {true_label}")
    print(f"Signal Vector Length: {len(signal_vector)} samples")

    # Predict using default best model
    res = predict_single_beat(signal_vector, model_name="best_model.pkl")

    print("\n[PREDICTION RESULT]:")
    print(f"  - Model Used: {res['model_used']}")
    print(f"  - Predicted Class: {res['predicted_class']}")
    print(f"  - Class Description: {res['class_description']}")
    print(f"  - Confidence: {res['confidence']}")
    print(f"  - Probabilities: {res['class_probabilities']}")
    print(f"  - Disclaimer: {res['disclaimer']}")

    assert res['predicted_class'] is not None, "Error: Prediction failed!"
    print("\n" + "=" * 60)
    print("  SUCCESS: Prediction Module Passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_prediction()
