import os
import ast
import glob
import wfdb
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================
# CONFIGURATION
# =====================
PTBXL_DATA_DIR = r"D:\New folder (2)\project day\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ARRHYTHMIA_DATA_PATH = r"D:\New folder (2)\project day\arrhythmia.data.csv"
PTBDB_DIR=r"D:\New folder (2)\project day\ptb-diagnostic-ecg-database-1.0.0\ptb-diagnostic-ecg-database-1.0.0"
CHAPMAN_DIR=r"D:\New folder (2)\project day\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"

MAX_PTB_RECORDS = None 
RFE_FEATURES = 150
RANDOM_STATE = 42

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================
# FEATURE EXTRACTION (with bandpass filtering + normalization)
# =====================
def bandpass_filter(signal, fs=500, lowcut=0.5, highcut=40, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
def extract_advanced_features(signal, fs=500):
    features = []
    for lead in signal.T:
        # Apply bandpass filter
        try:
            lead = bandpass_filter(lead, fs=fs)
        except Exception:
            continue

        # Normalize
        if np.std(lead) > 0:
            lead = (lead - np.mean(lead)) / np.std(lead)

        # Basic statistics
        features.extend([
            np.mean(lead), np.std(lead),
            np.min(lead), np.max(lead),
            skew(lead), kurtosis(lead) ])

        # R-peak detection + RR intervals
        r_peaks, _ = find_peaks(lead, distance=int(0.25 * fs))
        rr_intervals = np.diff(r_peaks) / fs
        if len(rr_intervals) > 0:
            features.extend([np.mean(rr_intervals), np.std(rr_intervals),np.min(rr_intervals), np.max(rr_intervals)])
        else:
            features.extend([np.nan] * 4)
    return features

# =====================
# DATA LOADING HELPERS
# =====================
def load_ptbxl_metadata(ptbxl_dir):
    meta_path = os.path.join(ptbxl_dir, "ptbxl_database.csv")
    if not os.path.exists(meta_path):
        logging.error(f"Metadata file not found at: {meta_path}")
        raise FileNotFoundError(meta_path)
    return pd.read_csv(meta_path)

def parse_scp_codes(scp_str):
    try:
        scp_dict = ast.literal_eval(scp_str)
        if isinstance(scp_dict, dict) and scp_dict:
            return max(scp_dict.items(), key=lambda x: x[1])[0]
    except Exception:
        pass
    return None

def find_dat_files(ptbxl_dir, suffix='_hr'):
    dat_files = glob.glob(os.path.join(ptbxl_dir, "records500", "**", f"*{suffix}.dat"), recursive=True)
    dat_files += glob.glob(os.path.join(ptbxl_dir, "records1000", "**", f"*{suffix}.dat"), recursive=True)

    dat_map = {}
    for f in dat_files:
        fname = os.path.splitext(os.path.basename(f))[0]
        ecg_num_str = fname.split('_')[0]
        try:
            ecg_num = int(ecg_num_str)
            dat_map[ecg_num] = f
        except ValueError:
            continue
    return dat_map

# =====================
# PTB-XL LOADING
# =====================
def load_ptbxl_features_and_labels(ptb_meta_df, dat_file_map):
    features_list, labels_list = [], []
    for _, row in tqdm(ptb_meta_df.iterrows(), total=len(ptb_meta_df), desc="PTB-XL"):
        ecg_id = row.get('ecg_id')
        if ecg_id not in dat_file_map:
            continue
        label = parse_scp_codes(row['scp_codes'])
        if label is None:
            continue
        dat_path = dat_file_map[ecg_id]
        try:
            record = wfdb.rdrecord(dat_path[:-4])
            features = extract_advanced_features(record.p_signal)
            features_list.append(features)
            labels_list.append(label)  
        except Exception as e:
            logging.warning(f"Skipping record {ecg_id}: {e}")
            continue
    return pd.DataFrame(features_list), pd.Series(labels_list)

# =====================
# ARRHYTHMIA LOADING
# =====================
def load_arrhythmia_data(arr_path):
    arr_data = pd.read_csv(arr_path, header=None, na_values='?').apply(pd.to_numeric, errors='coerce')
    arr_data.fillna(arr_data.mean(), inplace=True)
    X = arr_data.iloc[:, :-1]
    y = arr_data.iloc[:, -1].apply(lambda val: 0 if val == 1 else 1)
    return X, y

# =====================
# PTB-DB LOADING
# =====================
def load_ptbdb_dataset(ptbdb_dir):
    dat_files = glob.glob(os.path.join(ptbdb_dir, "patient*/*.dat"))
    features, labels = [], []
    
    for dat_file in tqdm(dat_files, desc="PTB-DB"):
        hea_file = dat_file.replace(".dat", ".hea")
        try:
            # Extract ECG features
            rec = wfdb.rdrecord(dat_file[:-4])
            sig = rec.p_signal
            features.append(extract_advanced_features(sig))
            
            # Assign label from .hea
            with open(hea_file, "r") as f:
                header = f.read().upper()
                if "HEALTHY CONTROL" in header:
                    labels.append(0)  # Normal
                else:
                    labels.append(1)  # Arrhythmia / Disease
        except Exception as e:
            logging.warning(f"Failed processing {dat_file}: {e}")
            continue
    
    return pd.DataFrame(features), pd.Series(labels)

# =====================
# CHAPMAN DATASET LOADING (.mat + .hea)
# =====================
def load_chapman_dataset(CHAPMAN_DIR):
    mat_files = glob.glob(os.path.join(CHAPMAN_DIR, "*.mat"))
    features, labels = [], []
    for mat_file in tqdm(mat_files, desc=" Chapman"):
        try:
            # Load ECG signal from .mat
            mat_data = loadmat(mat_file)
            if "val" not in mat_data:
                continue
            sig = np.array(mat_data["val"]).T  # shape: (time, leads)

            # Feature extraction
            features.append(extract_advanced_features(sig))

            # Corresponding header file
            hea_file = mat_file.replace(".mat", ".hea")
            if os.path.exists(hea_file):
                with open(hea_file, "r") as f:
                    header = f.read().upper()
                    labels.append(0 if "NORMAL" in header else 1)
            else:
                labels.append(1)  # default to arrhythmia if no header
        except Exception as e:
            logging.warning(f"Failed processing {mat_file}: {e}")
            continue
    df_X, df_y = pd.DataFrame(features), pd.Series(labels)
    logging.info(f"Chapman dataset loaded: {df_X.shape}, labels: {len(df_y)}")
    return df_X, df_y

# =====================
# MODEL EVALUATION
# =====================
def print_evaluation_metrics(y_true, y_pred, model_name):
    logging.info(f"--- {model_name} ---")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    logging.info(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    logging.info(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

def plot_confusion_matrix(y_true, y_pred, labels, title, cmap, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_external_dataset(X_ext, y_ext, model, imputer, scaler, rfe, name, save_csv=True):
    X_ext = X_ext.reindex(columns=range(imputer.n_features_in_), fill_value=np.nan)
    
    #Preprocessing
    X_ext_imputed = imputer.transform(X_ext)
    X_ext_scaled = scaler.transform(X_ext_imputed)
    X_ext_rfe = rfe.transform(X_ext_scaled)

    #Predictions
    y_pred = model.predict(X_ext_rfe)
    y_proba = model.predict_proba(X_ext_rfe)[:, 1]

    #Metrics
    logging.info(f"\n=== External Validation: {name} ===")
    print_evaluation_metrics(y_ext,y_pred, f"External{name}")

    #Confusion Matrix
    cm = confusion_matrix(y_ext, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")
    plot_confusion_matrix(y_ext,y_pred,["Normal", "Arrhythmia"], f"External{name}",'Reds',save_path=f"cm_{name.lower().replace('','-')}.png")
    if save_csv:
        df_out = pd.DataFrame({"record": range(len(y_ext)), "true_label": y_ext, "pred_label": y_pred, "arrhythmia_prob": y_proba})
        csv_file = f"predictions_{name.lower().replace(' ', '_')}.csv"
        df_out.to_csv(csv_file, index=False)
        logging.info(f"Predictions saved → {csv_file}")

# =====================
# MAIN PIPELINE
# =====================
def run_pipeline():

    # Load PTB-XL data
    ptb_meta_df = load_ptbxl_metadata(PTBXL_DATA_DIR)
    dat_file_map = find_dat_files(PTBXL_DATA_DIR)
    ptb_features_df, ptb_labels = load_ptbxl_features_and_labels(ptb_meta_df, dat_file_map)
    ptb_labels_binary = ptb_labels.apply(lambda x: 0 if x == 'NORM' else 1)
    ptb_features_df.fillna(ptb_features_df.mean(), inplace=True)

    # Load Arrhythmia data
    arr_X, arr_y = load_arrhythmia_data(ARRHYTHMIA_DATA_PATH)

    # Combine datasets
    X = pd.concat([ptb_features_df, arr_X], ignore_index=True)
    y = pd.concat([ptb_labels_binary, arr_y], ignore_index=True)
    X.fillna(X.mean(), inplace=True)
    logging.info(f"Combined dataset shape: {X.shape}, labels: {y.shape}")
    logging.info(f"PTB-XL records used: {len(ptb_features_df)}")
    logging.info(f"Arrhythmia records used: {len(arr_X)}")
    logging.info(f"Total combined records: {len(X)}")

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_res, y_train_res = smote.fit_resample(x_train_scaled, y_train)

    # Feature selection-----RFE------
    lr_rfe = LogisticRegression(solver='saga', penalty='l2', max_iter=20000, random_state=RANDOM_STATE)
    rfe = RFE(lr_rfe, n_features_to_select=RFE_FEATURES, step=5)
    rfe.fit(x_train_res, y_train_res)
    x_train_rfe = rfe.transform(x_train_res)
    x_test_rfe = rfe.transform(x_test_scaled)

   # Logistic Regression
    lr = LogisticRegression(solver='saga', penalty='l2', max_iter=10000, random_state=42)
    lr.fit(x_train_rfe, y_train_res)
    y_pred_lr = lr.predict(x_test_rfe)
    print_evaluation_metrics(y_test, y_pred_lr, "Logistic Regression")
    plot_confusion_matrix(y_test, y_pred_lr, ["Normal", "Arrhythmia"], "Logistic Regression", 'Blues')

    # Random Forest with tuning
    param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [4, 6, 8, 10, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None] }
    
    rf = RandomForestClassifier(random_state=42)
    rs = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1,verbose=2)
    rs.fit(x_train_rfe, y_train_res)
    rf_best = rs.best_estimator_
    y_pred_rf = rf_best.predict(x_test_rfe)
    print_evaluation_metrics(y_test, y_pred_rf, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_rf, ["Normal", "Arrhythmia"], "Random Forest", 'Greens')

    # XGBoost
    xgb_model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',use_label_encoder=False,random_state=RANDOM_STATE,n_jobs=-1)
    xgb_model.fit(x_train_rfe, y_train_res)
    y_pred = xgb_model.predict(x_test_rfe)
    print_evaluation_metrics(y_test, y_pred, "XGBoost")
    plot_confusion_matrix(y_test, y_pred, ["Normal", "Arrhythmia"], "XGBoost", 'Purples')

    # Save model and preprocessing pipeline
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(rfe, "rfe.pkl")
    logging.info("✅ Model and pipeline saved successfully.")
    
    # External validation
    ptbd_X, ptbd_y=load_ptbdb_dataset(PTBDB_DIR)
    chap_X, chap_y=load_chapman_dataset(CHAPMAN_DIR)
    model, imputer, scaler, rfe = joblib.load("xgb_model.pkl"),joblib.load("imputer.pkl"),joblib.load("scaler.pkl"),joblib.load("rfe.pkl")
    evaluate_external_dataset(ptbd_X, ptbd_y,model, imputer, scaler, rfe,"PTB-DB")
    evaluate_external_dataset(chap_X, chap_y,model, imputer, scaler, rfe,"Chapman-Shaoxing")
if __name__ == "__main__":
    run_pipeline()






