"""
Flask Web Application for ECG Arrhythmia Detection & Classification.
Provides HTTP routes for project dashboard, dataset analysis, ML model comparison,
static visualizations, and real-time heartbeat prediction.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.prediction import predict_single_beat, load_model, load_scaler
from src.data_loader import AAMI_CLASS_NAMES


app = Flask(__name__)
app.secret_key = "ecg_arrhythmia_detection_secret_key_2026"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), "static", "uploads")

ALLOWED_EXTENSIONS = {'csv', 'txt'}

BASE_DIR = os.path.dirname(__file__)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_evaluation_data():
    """
    Loads saved evaluation metrics from data/processed/.
    """
    csv_path = os.path.join(PROCESSED_DATA_DIR, "model_comparison.csv")
    json_path = os.path.join(PROCESSED_DATA_DIR, "model_evaluation_results.json")
    
    comp_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    
    eval_json = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            eval_json = json.load(f)
            
    return comp_df, eval_json


# =========================================================================
# ROUTES
# =========================================================================

@app.route("/")
def index():
    """Home Dashboard Page."""
    comp_df, eval_json = load_evaluation_data()
    
    total_samples = eval_json.get('total_samples', 109468)
    train_samples = eval_json.get('train_samples', 83111)
    test_samples = eval_json.get('test_samples', 26357)
    best_model = eval_json.get('best_model', 'Weighted KNN')
    num_classes = len(eval_json.get('classes', ['N', 'S', 'V', 'F', 'Q']))
    
    return render_template(
        "index.html",
        total_samples=total_samples,
        train_samples=train_samples,
        test_samples=test_samples,
        best_model=best_model,
        num_classes=num_classes
    )


@app.route("/about")
def about():
    """About & Project Documentation Page."""
    return render_template("about.html")


@app.route("/models")
def models():
    """Model Comparison Page."""
    comp_df, eval_json = load_evaluation_data()
    models_table = comp_df.to_dict(orient="records") if not comp_df.empty else []
    best_model = eval_json.get('best_model', 'Weighted KNN')
    
    return render_template(
        "models.html",
        models_table=models_table,
        best_model=best_model,
        eval_json=eval_json
    )


@app.route("/visualizations")
def visualizations():
    """Visualizations Gallery Page."""
    return render_template("visualizations.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """ECG Arrhythmia Prediction Interface & POST Handler."""
    if request.method == "GET":
        return render_template("predict.html", result=None, class_names=AAMI_CLASS_NAMES)

    # POST Request - Perform Prediction
    try:
        signal_vector = []
        source_info = ""

        # Option A: Vector passed in JSON / Form input
        if request.is_json:
            data = request.get_json()
            signal_vector = data.get("signal_vector", [])
            model_choice = data.get("model_name", "best_model.pkl")
            source_info = "API Input"
        elif "signal_data" in request.form and request.form["signal_data"].strip():
            raw_str = request.form["signal_data"].strip()
            # Parse comma or space separated values
            raw_str = raw_str.replace("[", "").replace("]", "").replace("\n", ",")
            signal_vector = [float(x.strip()) for x in raw_str.split(",") if x.strip()]
            model_choice = request.form.get("model_name", "best_model.pkl")
            source_info = "Manual Input"

        # Option B: Uploaded CSV File
        elif "file" in request.files:
            file = request.files["file"]
            model_choice = request.form.get("model_name", "best_model.pkl")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read CSV
                df_file = pd.read_csv(filepath, header=None)
                first_row = df_file.iloc[0].values.tolist()
                
                # If last item is label string/int, exclude
                if len(first_row) == 181:
                    first_row = first_row[:-1]
                    
                signal_vector = [float(x) for x in first_row if not pd.isna(x)]
                source_info = f"Uploaded File: {filename}"
            else:
                flash("Invalid file format. Please upload a .csv file containing 180 ECG sample points.", "danger")
                return render_template("predict.html", result=None, class_names=AAMI_CLASS_NAMES)
        else:
            flash("Please provide valid ECG signal vector data or upload a CSV file.", "warning")
            return render_template("predict.html", result=None, class_names=AAMI_CLASS_NAMES)

        # Validate vector length
        if len(signal_vector) != 180:
            error_msg = f"Invalid ECG signal length. Expected 180 sample points, but received {len(signal_vector)} points."
            if request.is_json:
                return jsonify({"status": "error", "message": error_msg}), 400
            flash(error_msg, "danger")
            return render_template("predict.html", result=None, class_names=AAMI_CLASS_NAMES)

        # Perform prediction using src/prediction.py
        result = predict_single_beat(signal_vector, model_name=model_choice)
        result["source_info"] = source_info
        result["input_signal"] = signal_vector

        if request.is_json:
            return jsonify({"status": "success", "result": result})

        return render_template("predict.html", result=result, class_names=AAMI_CLASS_NAMES)

    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        if request.is_json:
            return jsonify({"status": "error", "message": error_msg}), 500
        flash(error_msg, "danger")
        return render_template("predict.html", result=None, class_names=AAMI_CLASS_NAMES)


@app.route("/api/sample-beats", methods=["GET"])
def get_sample_beats():
    """API endpoint to get sample heartbeats for each AAMI class for demo prediction."""
    csv_path = os.path.join(PROCESSED_DATA_DIR, "processed_beats.csv")
    if not os.path.exists(csv_path):
        return jsonify({"status": "error", "message": "Processed dataset not found."}), 404

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith('s_')]
    
    samples = {}
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        subset = df[df['aami_class'] == cls]
        if not subset.empty:
            sample_row = subset.iloc[0]
            vector = sample_row[feature_cols].values.tolist()
            samples[cls] = {
                'class': cls,
                'record_id': str(sample_row['record_id']),
                'class_name': AAMI_CLASS_NAMES.get(cls, cls),
                'signal': [round(float(x), 4) for x in vector]
            }

    return jsonify({"status": "success", "samples": samples})


# Custom Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", error_title="404 - Page Not Found", error_msg="The requested page could not be found."), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("base.html", error_title="500 - Internal Server Error", error_msg="An internal application error occurred."), 500


if __name__ == "__main__":
    print("Starting ECG Arrhythmia Classification Flask Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)
