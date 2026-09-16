# Machine Learning Model Training & Evaluation Documentation

## 1. Features & Input Representation
* **Signal Representation:** Each heartbeat is represented as a digitized 180-sample amplitude vector ($90$ samples prior to R-peak, $90$ samples following R-peak).
* **Feature Dimension:** 180 continuous numerical features (`s_0` through `s_179`).
* **Feature Scaling:** `StandardScaler` (Z-score scaling) fit strictly on training set $X_{\text{train}}$ and stored as `models/scaler.pkl`.

## 2. Target Classes (AAMI EC57 Standard)
* `N`: Normal / Bundle Branch Block Beat
* `S`: Supraventricular Ectopic Beat (SVEB)
* `V`: Ventricular Ectopic Beat (VEB)
* `F`: Fusion Beat
* `Q`: Paced / Unknown / Unclassifiable Beat

## 3. Train-Test Split & Data Leakage Prevention
To prevent data leakage between beats belonging to the same patient record:
* **Split Algorithm:** Group-based split (`GroupShuffleSplit` on `record_id`).
* **Split Ratio:** 36 Training Records (~75%) / 12 Testing Records (~25%).
* **Train Samples:** `83,111` heartbeats
* **Test Samples:** `26,357` heartbeats
* **Random Seed:** `42` for strict reproducibility.

## 4. Evaluated Machine Learning Algorithms
1. **Logistic Regression:** Multinomial, L-BFGS solver, max_iter=1000.
2. **K-Nearest Neighbors (KNN):** $k=5$, uniform distance weights, KD-tree algorithm.
3. **Weighted KNN:** $k=5$, inverse-distance weights.
4. **Decision Tree:** Criterion=Gini, max_depth=15.
5. **Random Forest:** 100 trees, max_depth=20, random_state=42.
6. **Naive Bayes:** GaussianNB.
7. **SVM:** Calibrated Linear SGD-SVM with hinge loss & L2 penalty.

## 5. Model Evaluation Results & Performance Metrics
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | **0.8983** | 0.5056 | 0.3800 | 0.3691 | 0.8907 |
| **Logistic Regression** | 0.8542 | 0.4485 | 0.3704 | 0.3747 | 0.8698 |
| **Random Forest** | 0.8542 | 0.5186 | 0.3888 | 0.4109 | **0.8742** |
| **K-Nearest Neighbors (KNN)** | 0.8363 | 0.4578 | 0.4705 | 0.4483 | 0.8718 |
| **Weighted KNN** | 0.8351 | 0.4578 | **0.4716** | **0.4490** | 0.8712 |
| **Decision Tree** | 0.8026 | 0.2854 | 0.2797 | 0.2732 | 0.8024 |
| **Naive Bayes** | 0.1556 | 0.2755 | 0.4124 | 0.2677 | 0.2025 |

*Selected Best Model by Macro F1:* **Weighted KNN** (Saved to `models/best_model.pkl`).

## 6. Model Artifacts Saved
All trained models are serialized using `joblib` into the `models/` directory:
- `models/logistic_regression.pkl`
- `models/knn.pkl`
- `models/weighted_knn.pkl`
- `models/decision_tree.pkl`
- `models/random_forest.pkl`
- `models/naive_bayes.pkl`
- `models/svm.pkl`
- `models/best_model.pkl`
- `models/scaler.pkl`

## 7. Limitations & Academic Disclaimer
* **Patient Inter-Subject Variability:** Record-based split tests generalize models to unseen patient records. Patient-specific ECG morphologies cause variation across testing records.
* **Academic Project Disclaimer:** This application is developed for educational and academic research purposes only and is not a certified medical diagnostic system.
