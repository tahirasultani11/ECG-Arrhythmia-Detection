
# ECG-Arrhythmia-Detection

ECG-Arrhythmia-Detection is an open-source project for *automated arrhythmia classification* using multiple ECG datasets. It integrates advanced *feature extraction techniques, robust **machine learning models* (Logistic Regression, Random Forest, and XGBoost), and validates across external benchmark datasets (PTB-XL, PTB-DB, Chapman-Shaoxing).

---

## 🌟 Model in Action

Imagine a cardiologist reviewing ECG signals. This pipeline streamlines the workflow:

1. *Signal Preprocessing:* Raw ECG signals undergo bandpass filtering (0.5–40Hz) and normalization.
2. *Feature Extraction:* Extracts statistical features, R-peak detection, and RR-interval features.
3. *Dataset Integration:* Combines *PTB-XL* (large ECG dataset) with *Arrhythmia dataset* for training.
4. *Model Training:* Uses *Logistic Regression, Random Forest (with tuning), and XGBoost*.
5. *External Validation:* Evaluates trained models on *PTB-DB* and *Chapman-Shaoxing* datasets.
6. *Performance Insights:* Reports metrics (Accuracy, Precision, Recall, F1) with confusion matrices.

---

## ✨ Features

* *Advanced Signal Preprocessing:* Bandpass filtering, normalization, and feature extraction.
* *Multi-Dataset Integration:* Supports PTB-XL, Arrhythmia, PTB-DB, and Chapman-Shaoxing.
* *Imbalanced Data Handling:* Uses *SMOTE* oversampling for balanced learning.
* *Feature Selection:* Recursive Feature Elimination (RFE) with Logistic Regression.
* *Model Variety:* Logistic Regression, Random Forest (with RandomizedSearchCV), and XGBoost.
* *External Generalization:* Validates model robustness on external ECG datasets.
* *Evaluation Tools:* Confusion matrices and metrics logging.
* *Model Saving:* Saves trained pipelines with Joblib.

---

## 🛠 Built With

*Core Libraries & Tools:*

* [WFDB](https://wfdb.readthedocs.io/) - ECG signal reading
* [SciPy](https://scipy.org/) - Signal processing and feature extraction
* [Scikit-learn](https://scikit-learn.org/) - ML pipeline, feature selection, evaluation
* [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting classifier
* [Imbalanced-learn](https://imbalanced-learn.org/) - SMOTE oversampling
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.8+
* Install required dependencies:

  bash
  pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn wfdb xgboost tqdm joblib
  

### Dataset Setup

1. *Download and Extract Datasets:*

   * [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
   * [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
   * [Chapman-Shaoxing ECG Dataset](https://figshare.com/collections/ChapmanECG/4560497)
   * [Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/arrhythmia)

2. *Update Directory Paths:*
   Inside the script, update dataset paths:

   python
   PTBXL_DATA_DIR = r"your_path/ptb-xl"
   ARRHYTHMIA_DATA_PATH = r"your_path/arrhythmia.data.csv"
   PTBDB_DIR = r"your_path/ptbdb"
   CHAPMAN_DIR = r"your_path/chapman"
   

### Usage

1. Clone this repository:

   bash
   git clone https://github.com/tahirasultani11/ECG-Arrhythmia-Detection.git
   cd ECG-Arrhythmia-Detection
   

2. Run the pipeline:

   bash
   python main.py
   

3. Outputs:

   * Trained models saved as xgb_model.pkl, imputer.pkl, scaler.pkl, rfe.pkl
   * Confusion matrices saved as .png
   * Predictions stored in .csv

---

## 📊 Example Results

The pipeline reports metrics for each model:

* *Logistic Regression:* Fast baseline model
* *Random Forest:* Robust with tuned hyperparameters
* *XGBoost:* High-performance boosting classifier

External validation ensures generalizability across multiple datasets.

---
