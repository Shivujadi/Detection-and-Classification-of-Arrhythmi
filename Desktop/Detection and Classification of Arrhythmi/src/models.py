"""
Machine Learning Models Module for ECG Arrhythmia Classification.
Defines model initialization, training pipelines, evaluation metrics,
scaling, serialization, and comparison functions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# Fixed Random Seed for Reproducibility
RANDOM_SEED = 42

# AAMI 5-Class Labels
CLASSES = ['N', 'S', 'V', 'F', 'Q']


def load_dataset(data_path=None):
    """
    Loads processed heartbeat dataset and extracts features, targets, and group record IDs.
    """
    if data_path is None:
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_beats.csv")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run Phase 4 preprocessing first.")

    df = pd.read_csv(data_path)
    
    # Feature columns s_0 to s_179
    feature_cols = [c for c in df.columns if c.startswith('s_')]
    
    X = df[feature_cols].values
    y = df['aami_class'].values
    groups = df['record_id'].astype(str).values
    
    return X, y, groups, feature_cols, df


def split_data_by_record(X, y, groups, test_size=0.25, random_state=RANDOM_SEED):
    """
    Splits heartbeats into Train and Test sets based on patient record IDs (GroupShuffleSplit).
    Guarantees ZERO data leakage between beats from the same record.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    
    train_records = sorted(list(set(groups_train)))
    test_records = sorted(list(set(groups_test)))
    
    print(f"Group-Based Split Complete:")
    print(f"  - Training Records ({len(train_records)}): {train_records}")
    print(f"  - Testing Records ({len(test_records)}): {test_records}")
    print(f"  - Train Heartbeats: {len(X_train)}")
    print(f"  - Test Heartbeats: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, train_records, test_records


def train_scaler(X_train):
    """
    Fits StandardScaler on X_train only and saves it.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to: {scaler_path}")
    
    return scaler, X_train_scaled


def get_model_definitions():
    """
    Returns a dictionary of all 7 machine learning models to evaluate.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED, n_jobs=-1
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='uniform', n_jobs=-1
        ),
        'Weighted KNN': KNeighborsClassifier(
            n_neighbors=5, weights='distance', n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15, criterion='gini', random_state=RANDOM_SEED
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'SVM': CalibratedClassifierCV(
            SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=RANDOM_SEED),
            cv=3
        )
    }
    return models


def evaluate_model(model, X_test, y_test, labels=CLASSES):
    """
    Evaluates a trained model on X_test and returns metrics dictionary.
    """
    y_pred = model.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average='macro', zero_division=0
    )
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    
    return {
        'accuracy': round(acc, 4),
        'precision_macro': round(float(p_macro), 4),
        'recall_macro': round(float(r_macro), 4),
        'f1_macro': round(float(f1_macro), 4),
        'precision_weighted': round(float(p_weighted), 4),
        'recall_weighted': round(float(r_weighted), 4),
        'f1_weighted': round(float(f1_weighted), 4),
        'confusion_matrix': cm
    }
