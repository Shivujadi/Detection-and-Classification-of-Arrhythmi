"""
Prediction Module for ECG Arrhythmia Classification.
Provides reusable functions for loading trained models and performing
predictions on single or batch ECG heartbeat signal vectors.
"""

import os
import joblib
import numpy as np

from src.data_loader import AAMI_CLASS_NAMES
from src.preprocessing import normalize_heartbeat, butter_bandpass_filter


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_model(model_name="best_model.pkl"):
    """
    Loads a saved joblib model from models/ directory.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        # Fallback to random_forest.pkl if best_model.pkl does not exist yet
        alt_path = os.path.join(MODELS_DIR, "random_forest.pkl")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Trained model file not found at: {model_path}")
            
    return joblib.load(model_path)


def load_scaler():
    """
    Loads the fitted StandardScaler from models/scaler.pkl.
    """
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    return joblib.load(scaler_path)


def predict_single_beat(signal_vector, model_name="best_model.pkl"):
    """
    Predicts the arrhythmia class for a single 180-sample ECG heartbeat vector.
    
    Args:
        signal_vector (list or np.ndarray): 180 numerical amplitude points.
        model_name (str): Filename of the trained model to use.
        
    Returns:
        dict: Prediction results including predicted_class, class_name,
              probabilities (if supported), and disclaimer.
    """
    signal_arr = np.array(signal_vector, dtype=float)
    if len(signal_arr) != 180:
        raise ValueError(f"Expected 180 ECG sample points, but got {len(signal_arr)} samples.")

    # 1. Normalize segment
    norm_signal = normalize_heartbeat(signal_arr)
    
    # 2. Scale features using stored StandardScaler
    scaler = load_scaler()
    scaled_signal = scaler.transform(norm_signal.reshape(1, -1))
    
    # 3. Load model and predict
    model = load_model(model_name)
    pred_class = model.predict(scaled_signal)[0]
    
    # 4. Probabilities / Confidence
    probabilities = None
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(scaled_signal)[0]
            classes = model.classes_
            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            probabilities = prob_dict
            confidence = float(np.max(probs))
        except Exception:
            probabilities = None

    class_fullname = AAMI_CLASS_NAMES.get(pred_class, str(pred_class))

    return {
        'model_used': model_name,
        'predicted_class': str(pred_class),
        'class_description': class_fullname,
        'confidence': round(confidence, 4) if confidence else None,
        'class_probabilities': probabilities,
        'disclaimer': (
            "This application is developed for academic and educational purposes only. "
            "It is not a medical diagnostic tool and should not be used for clinical decisions."
        )
    }
