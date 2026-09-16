# Flask Web Application Documentation

## 1. Application Overview
The Flask web application provides an interactive web dashboard for real-time ECG arrhythmia classification, model performance comparison, dataset visualizations, and academic project demonstration.

## 2. Startup Commands
To launch the Flask web application server:

```bash
# Activate virtual environment and run app.py
./venv/bin/python app.py
```
*Access in browser at:* `http://127.0.0.1:5000`

## 3. Available Application Routes
| HTTP Method | Route URL | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Home Dashboard showing dataset metrics, project overview, and pipeline diagram. |
| `GET` | `/about` | Project background, problem statement, MIT-BIH database, and AAMI standard. |
| `GET` | `/models` | Interactive table displaying calculated evaluation metrics across all 7 trained models. |
| `GET` | `/visualizations` | Gallery displaying ECG signal waveforms, confusion matrices, and feature importance. |
| `GET` | `/predict` | Interactive prediction form for uploading CSV files or selecting demo sample beats. |
| `POST` | `/predict` | Endpoint handling JSON/Form prediction requests for a 180-sample heartbeat vector. |
| `GET` | `/api/sample-beats` | API returning pre-extracted sample heartbeats for each AAMI class from MIT-BIH DB. |

## 4. Expected Input Format
* **Vector Dimension:** 180 continuous amplitude sample points (`s_0` through `s_179`).
* **Normal Range:** Min-Max normalized values in range $[0, 1]$.
* **Input Sources:**
  1. Interactive sample beat dropdown (fetches real MIT-BIH beats from `/api/sample-beats`).
  2. Manual comma-separated array input.
  3. Uploaded CSV file containing 180 sample columns.

## 5. Trained Model Integration
* Default prediction model: `models/best_model.pkl` (Weighted KNN with Macro F1 = 0.4490).
* Alternative selectable models: Random Forest, KNN, Logistic Regression, Calibrated SVM, Decision Tree, Naive Bayes.
* Feature Scaling: Automatically applies stored `models/scaler.pkl` (`StandardScaler`).

## 6. Academic Disclaimer & Limitations
* **Academic Research Purpose:** Developed exclusively for educational and academic research.
* **Not a Medical Device:** The system is not clinically validated for patient diagnostic decisions.
