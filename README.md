#🫀 ECG Arrhythmia Classification Pipeline

This repository contains a machine learning pipeline for classifying ECG signals into Normal vs Arrhythmia across multiple publicly available datasets. It integrates feature extraction, preprocessing, model training, and external validation in a robust and reproducible way.

---

##📂 Supported Datasets

1. PTB-XL – Large-scale annotated ECG dataset.


2. PTB Diagnostic Database (PTB-DB) – Smaller dataset with detailed clinical ECGs.


3. Chapman-Shaoxing – Large-scale 12-lead ECG dataset (.mat + .hea format).


4. UCI Arrhythmia Dataset – Tabular arrhythmia dataset.

---

⚙ Pipeline Overview

1. Feature Extraction

Performed on ECG signals (all leads).

Includes:

Bandpass filtering (0.5–40 Hz) → removes baseline drift & noise.

Normalization (zero mean, unit variance per lead).

Statistical features (mean, std, min, max, skewness, kurtosis).

Heart-rate variability (RR intervals from R-peaks).

2. Data Preprocessing

Missing values handled with mean imputation.

Features scaled with StandardScaler.

Imbalanced data handled with SMOTE oversampling.

3. Feature Selection

Recursive Feature Elimination (RFE) with Logistic Regression.

Selects top 150 most informative features.

4. Models

Logistic Regression (baseline).

Random Forest (with RandomizedSearchCV tuning).

XGBoost (final model).

5. Evaluation

Performance metrics: Accuracy, Precision, Recall, F1 Score.

Confusion Matrix visualization with Seaborn.

External validation on PTB-DB and Chapman-Shaoxing.

Predictions saved as .csv for leaderboard submission.

---

📊 Results

Consistent performance across datasets due to robust preprocessing.

External validation ensures generalization, not just overfitting to PTB-XL.

XGBoost model achieves the best tradeoff between accuracy and generalization.

---

🛠 Installation

# Clone repo
git clone https://github.com/tahirasultani11/ECG-Arrhythmia-Detection.git
cd ecg-arrhythmia-classification
# Install dependencies
pip install -r requirements.tx

---

▶ Usage

Train + Validate

python main.py

Outputs

Trained models and preprocessing pipeline:

xgb_model.pkl, scaler.pkl, imputer.pkl, rfe.pkl


Confusion matrix plots:

cm_logistic_regression.png, cm_random_forest.png, cm_xgboost.png, etc.


External validation predictions:

predictions_ptb-db.csv, predictions_chapman-shaoxing.csv

---

🧩 Repository Structure

.
├── main.py                     
├── requirements.txt           
├── README.md                   
├── data/                       
│   ├── ptb-xl/
│   ├── ptbdb/
│   ├── chapman-shaoxing/
│   └── arrhythmia.csv
└── outputs/                    


📌 Future Improvements

Add deep learning models (CNN/LSTM) for raw ECG.

Cache pre-extracted features for faster runs.

Incorporate more advanced HRV features.



---

✨ Acknowledgments

PhysioNet for PTB-XL and PTB-DB.

Chapman University & Shaoxing People’s Hospital for Chapman-Shaoxing dataset.

UCI Machine Learning Repository for Arrhythmia dataset.
