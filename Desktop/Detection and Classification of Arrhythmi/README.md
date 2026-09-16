# Detection and Classification of Arrhythmia Using Machine Learning

## 1. Project Overview
This project is an end-to-end Machine Learning system for detecting and classifying Cardiac Arrhythmia from ECG (Electrocardiogram) signals. It utilizes the PhysioNet MIT-BIH Arrhythmia Database, extracts 180-sample R-peak centered heartbeat vectors, trains 7 Machine Learning algorithms using group-based patient splitting (preventing data leakage), and serves a Flask web dashboard for real-time prediction and model evaluation.

## 2. Technologies Used
- **Language:** Python 3.14.7
- **Web Framework:** Flask 3.1.3
- **Signal Processing & ML:** WFDB, SciPy, Pandas, NumPy, Scikit-learn 1.9.1, Joblib
- **Data Visualization:** Matplotlib 3.11.2, Seaborn 0.13.2
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla ES6)

## 3. Quick Start & How to Run

### Step 1: Environment Setup
```bash
# Verify virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Data Preprocessing & Signal Segmentation
```bash
./venv/bin/python scripts/test_preprocessing.py
```
*(Processes 48 MIT-BIH records, applies Butterworth bandpass filtering, segments 109,468 heartbeats into `data/processed/processed_beats.csv`)*

### Step 3: Model Training & Evaluation
```bash
./venv/bin/python scripts/train_models.py
```
*(Trains 7 ML models using GroupShuffleSplit on patient record IDs, serializes `.pkl` models under `models/`, and generates comparison charts under `static/plots/`)*

### Step 4: Run Flask Web Application
```bash
./venv/bin/python app.py
```
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 4. Model Evaluation Summary Results
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM** | **0.8983** | 0.3908 | 0.3541 | 0.3691 | 0.8907 |
| **Logistic Regression** | 0.8542 | 0.3698 | 0.3913 | 0.3747 | 0.8698 |
| **Random Forest** | 0.8542 | 0.4333 | 0.4807 | 0.4109 | **0.8742** |
| **KNN** | 0.8363 | 0.4489 | 0.5522 | 0.4483 | 0.8718 |
| **Weighted KNN** *(Selected Best)* | 0.8351 | 0.4493 | **0.5563** | **0.4490** | 0.8712 |
| **Decision Tree** | 0.8026 | 0.3539 | 0.3091 | 0.2732 | 0.8024 |
| **Naive Bayes** | 0.1556 | 0.3809 | 0.4689 | 0.2677 | 0.2025 |

## 5. Project Directory Structure
```
Detection and Classification of Arrhythmi/
├── app.py                  # Flask web application entry point
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/
│   ├── raw/mitdb/          # Raw WFDB PhysioNet ECG records
│   └── processed/          # Processed beat features & metrics
├── models/                 # Saved model binaries (.pkl)
├── scripts/
│   ├── inspect_dataset.py  # Dataset inspection script
│   ├── generate_plots.py   # Plot figure generator
│   ├── train_models.py     # Master ML training pipeline
│   ├── test_preprocessing.py # Preprocessing test script
│   ├── test_prediction.py # Single beat prediction test
│   └── test_app.py         # Flask app test suite
├── src/
│   ├── data_loader.py     # WFDB record reader
│   ├── preprocessing.py   # Bandpass filter & heartbeat segmenter
│   ├── models.py          # Model definitions & group-split logic
│   └── prediction.py      # Reusable prediction engine
├── static/
│   ├── css/style.css      # Healthcare/ML web styling
│   ├── js/script.js        # Canvas waveform plotter & client script
│   └── plots/              # Saved PNG plot charts
├── templates/              # HTML Jinja2 templates
└── docs/                   # Architectural & viva documentation
```

## 6. Academic Disclaimer
This project is developed for educational and academic research purposes as part of a final-year engineering project. It is not a medical diagnostic tool and should not be used for clinical decision-making.
