"""
Machine Learning Model Training, Evaluation, and Plotting Pipeline.
Trains 7 ML models on processed MIT-BIH dataset with group-based record splitting,
evaluates metrics, serializes models, and generates comparison visualizations.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import (
    load_dataset, split_data_by_record, train_scaler,
    get_model_definitions, evaluate_model, CLASSES,
    MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
)
from src.data_loader import AAMI_CLASS_NAMES


PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "plots")


def run_training_pipeline():
    print("=" * 70)
    print("     PHASE 5: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[STEP 1]: Loading processed MIT-BIH dataset...")
    X, y, groups, feature_cols, df = load_dataset()
    
    print(f"Total Heartbeats: {len(X)}")
    print(f"Feature Vector Dimension: {X.shape[1]}")
    
    class_dist = df['aami_class'].value_counts().to_dict()
    print(f"Class Distribution: {class_dist}")

    # 2. Group-Based Patient Split
    print("\n[STEP 2]: Performing Patient/Record-Based Split (Preventing Data Leakage)...")
    X_train, X_test, y_train, y_test, train_recs, test_recs = split_data_by_record(
        X, y, groups, test_size=0.25, random_state=RANDOM_SEED
    )

    # 3. Fit & Apply StandardScaler
    print("\n[STEP 3]: Fitting StandardScaler on X_train...")
    scaler, X_train_scaled = train_scaler(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train & Evaluate Models
    print("\n[STEP 4]: Training and Evaluating 7 Machine Learning Models...")
    model_defs = get_model_definitions()
    
    # Filename mapping
    file_mapping = {
        'Logistic Regression': 'logistic_regression.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Weighted KNN': 'weighted_knn.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'SVM': 'svm.pkl'
    }

    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    for name, model in model_defs.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        metrics = evaluate_model(model, X_test_scaled, y_test, labels=CLASSES)
        results[name] = metrics
        
        # Save individual model pkl
        fname = file_mapping[name]
        m_path = os.path.join(MODELS_DIR, fname)
        joblib.dump(model, m_path)
        
        print(f"    -> Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1_macro']:.4f} | Weighted F1: {metrics['f1_weighted']:.4f}")
        
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_model_name = name
            best_model_obj = model

    # Save Best Model copy
    if best_model_obj:
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(best_model_obj, best_path)
        print(f"\nSelected Best Model: {best_model_name} (Saved to {best_path})")

    # 5. Save Comparison Results
    print("\n[STEP 5]: Saving Evaluation Results & Comparison Tables...")
    table_rows = []
    for m_name, m_metrics in results.items():
        table_rows.append({
            'Model': m_name,
            'Accuracy': m_metrics['accuracy'],
            'Macro Precision': m_metrics['precision_macro'],
            'Macro Recall': m_metrics['recall_macro'],
            'Macro F1 Score': m_metrics['f1_macro'],
            'Weighted Precision': m_metrics['precision_weighted'],
            'Weighted Recall': m_metrics['recall_weighted'],
            'Weighted F1 Score': m_metrics['f1_weighted']
        })
        
    comp_df = pd.DataFrame(table_rows)
    comp_csv_path = os.path.join(PROCESSED_DATA_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    
    summary_results = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_records': train_recs,
        'test_records': test_recs,
        'num_features': X.shape[1],
        'classes': CLASSES,
        'best_model': best_model_name,
        'results': results
    }
    json_path = os.path.join(PROCESSED_DATA_DIR, "model_evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"Saved evaluation comparison to: {comp_csv_path}")
    print(f"Saved JSON metrics to: {json_path}")

    # 6. Generate Plots
    print("\n[STEP 6]: Generating Visualization Plots...")
    generate_evaluation_plots(comp_df, results, best_model_obj, best_model_name, df)

    print("\n" + "=" * 70)
    print("  SUCCESS: Phase 5 Model Training & Evaluation Complete!")
    print("=" * 70)
    return summary_results


def generate_evaluation_plots(comp_df, results, best_model, best_model_name, df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Model Accuracy & F1 Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(comp_df))
    width = 0.35
    
    ax.bar(x - width/2, comp_df['Accuracy'], width, label='Accuracy', color='#2563eb')
    ax.bar(x + width/2, comp_df['Macro F1 Score'], width, label='Macro F1 Score', color='#059669')
    
    ax.set_title('Machine Learning Models Comparison — Accuracy vs Macro F1 Score', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Model'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=10)
    ax.legend(loc='lower right')
    
    for i in range(len(comp_df)):
        ax.text(i - width/2, comp_df['Accuracy'].iloc[i] + 0.02, f"{comp_df['Accuracy'].iloc[i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width/2, comp_df['Macro F1 Score'].iloc[i] + 0.02, f"{comp_df['Macro F1 Score'].iloc[i]:.2f}", ha='center', fontsize=8)
        
    plt.tight_layout()
    p1_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(p1_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # Plot 2: Best Model Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    cm = np.array(results[best_model_name]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix — Best Model ({best_model_name})', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted AAMI Class', fontsize=10)
    ax.set_ylabel('True AAMI Class', fontsize=10)
    plt.tight_layout()
    p2_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(p2_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2_path}")

    # Plot 3: Class Distribution Bar Plot
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    class_counts = df['aami_class'].value_counts()
    colors = ['#16a34a', '#ea580c', '#dc2626', '#9333ea', '#0284c7']
    
    bars = ax.bar([AAMI_CLASS_NAMES.get(c, c) for c in class_counts.index], class_counts.values, color=colors[:len(class_counts)])
    ax.set_title('MIT-BIH Processed Dataset Class Distribution (AAMI EC57)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Count (Heartbeats)', fontsize=10)
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels([f"{c}\n({count:,})" for c, count in zip(class_counts.index, class_counts.values)], fontsize=9)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
                    
    plt.tight_layout()
    p3_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    fig.savefig(p3_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p3_path}")

    # Plot 4: Random Forest Feature Importance (Top 20 Features)
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
            ax.bar(range(20), importances[indices], color='#2563eb')
            ax.set_title('Top 20 Important ECG Signal Features (Random Forest)', fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel('Sample Point Feature Index', fontsize=10)
            ax.set_ylabel('Importance Score', fontsize=10)
            ax.set_xticks(range(20))
            ax.set_xticklabels([f"s_{idx}" for idx in indices], rotation=45, fontsize=8)
            plt.tight_layout()
            p4_path = os.path.join(PLOTS_DIR, "feature_importance.png")
            fig.savefig(p4_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {p4_path}")


if __name__ == "__main__":
    run_training_pipeline()
