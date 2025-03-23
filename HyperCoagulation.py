# -*- coding: utf-8 -*-
"""
Main Steps:
1. Data Loading and Preprocessing (missing value handling, feature selection)
2. Splitting into training, internal testing, and external testing datasets
3. Oversampling/Undersampling (SMOTE)
4. Feature Selection
5. Data Standardization and Outlier Treatment
6. Model Definition and Parameter Optimization
7. Model Training, Evaluation, and Result Saving

Author: Peng Hu, Xiaojuan Xiong
Date: 2025-01-02
"""

import datetime
import logging
import threading

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# Global lock for multi-threaded SHAP plotting
lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------
# Evaluation Functions
# ---------------------------
def bootstrap_confidence_interval(y_true, y_scores, y_pred, metric_func, n_bootstrap=1000):
    """
    Compute the 95% confidence interval for a given metric using bootstrap sampling.
    """
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        y_true_resample, y_scores_resample, y_pred_resample = resample(y_true, y_scores, y_pred)
        bootstrap_sample = metric_func(y_true_resample, y_scores_resample, y_pred_resample)
        bootstrap_samples.append(bootstrap_sample)
    bootstrap_samples = np.array(bootstrap_samples)
    lower = np.percentile(bootstrap_samples, 2.5)
    upper = np.percentile(bootstrap_samples, 97.5)
    return lower, upper


def calculate_metrics(model, X_data, y_data, name):
    """
    Use bootstrap sampling to compute the mean and confidence interval of various evaluation metrics.
    Returns a DataFrame with metric information.
    """
    metrics_samples = {
        'auroc': [],
        'pr_auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f_score': []
    }

    # Perform bootstrap sampling 100 times
    for _ in range(100):
        X_resample, y_resample = resample(X_data, y_data)
        y_scores = model.predict_proba(X_resample)[:, 1]
        y_pred = model.predict(X_resample)

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_resample, y_scores)
        sort_idx = np.argsort(recall)
        precision = precision[sort_idx]
        recall = recall[sort_idx]

        # Compute metrics if there is more than one class
        if len(set(y_resample)) > 1:
            metrics_samples['auroc'].append(roc_auc_score(y_resample, y_scores))
        else:
            metrics_samples['auroc'].append(0)
        metrics_samples['pr_auc'].append(auc(recall, precision))
        metrics_samples['accuracy'].append(accuracy_score(y_resample, y_pred))
        metrics_samples['sensitivity'].append(recall_score(y_resample, y_pred))
        metrics_samples['specificity'].append(recall_score(1 - y_resample, 1 - y_pred))
        metrics_samples['precision'].append(precision_score(y_resample, y_pred, zero_division=1))
        metrics_samples['f_score'].append(f1_score(y_resample, y_pred))

    metrics_dict = {}
    for metric, samples in metrics_samples.items():
        metrics_dict[metric] = np.mean(samples)
        metrics_dict[f'{metric}_lower'] = np.percentile(samples, 2.5)
        metrics_dict[f'{metric}_upper'] = np.percentile(samples, 97.5)

    return pd.DataFrame(metrics_dict, index=[name])


def calculate_mcc_curve(y_val_prob, X_val, y_val):
    """
    Compute the MCC (Matthews Correlation Coefficient) curve data to assess model performance over different thresholds.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    mcc_scores = [matthews_corrcoef(y_val, y_val_prob >= t) for t in thresholds]
    return thresholds, mcc_scores


# ---------------------------
# Plotting Functions
# ---------------------------
def plot_mcc_curves(results, save_path):
    """
    Plot MCC curves for different models and save the figure.
    """
    plt.figure(figsize=(10, 6))
    for thresholds, mcc_scores, label in results:
        max_mcc = max(mcc_scores)
        plt.plot(thresholds, mcc_scores, label=f'{label} (Max MCC: {max_mcc:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title('MCC Curves for Different Models')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curves(pr_curves, title, save_path):
    """
    Plot the Precision-Recall curves.
    """
    plt.figure(figsize=(8, 6))
    for precision, recall, name, score in pr_curves:
        plt.plot(recall, precision, label=f'{name} ({score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curves(calibration_data, title, save_path):
    """
    Plot model calibration curves.
    """
    plt.figure(figsize=(8, 6))
    for fraction_of_positives, mean_predicted_value, name, score in calibration_data:
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name} (Brier: {score:.2f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_SHAP(name, model, X_train, selected_features):
    """
    Plot SHAP summary plot for the model; supports multi-threaded execution.
    """
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    logging.info(f"{datetime.datetime.now()} -- {name} -- start sampling for SHAP...")
    X_sample = shap.sample(X_train, nsamples=500)
    logging.info(f"{datetime.datetime.now()} -- {name} -- finished sampling")

    # Choose the appropriate SHAP explainer based on model type
    if name in ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    elif name == "Neural Networks":
        try:
            explainer = shap.DeepExplainer(model, X_train)
        except Exception:
            try:
                explainer = shap.KernelExplainer(model.predict, X_sample)
            except Exception:
                return
    elif name in ["Logistic Regression"]:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample)

    logging.info(f"{datetime.datetime.now()} -- {name} -- calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    with lock:
        plt.figure()
        shap.summary_plot(shap_values, X_sample, max_display=15, feature_names=selected_features, show=False)
        plt.savefig(f'./Death/Data/output/{name}-shap_plot-{now_str}.png', dpi=700)
        plt.close()


# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_data(path, encoding='UTF-8'):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(path, encoding=encoding)
        logging.info(f"Data loaded successfully from {path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        raise


def calculate_positive_rate(data, source_values, target_column):
    """
    Calculate the positive rate of the target variable for each source.
    """
    results = {}
    for source in source_values:
        source_data = data[data['source'] == source]
        positive_count = source_data[target_column].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        results[source] = (positive_rate, positive_count)
        logging.info(f"Source {source} - Positive Rate: {positive_rate:.2%}, Count: {positive_count}")
    return results


def handle_missing_values(data, missing_threshold=0.40):
    """
    Handle missing values in the dataset:
    1. Drop columns with missing ratio above the threshold.
    2. Fill binary variables with 0; impute continuous variables using KNNImputer; for categorical variables, encode, impute, then revert.
    """
    missing_ratio = data.isnull().sum() / len(data)
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    data = data.drop(columns_to_drop, axis=1)
    logging.info(f"Dropped columns with missing ratio > {missing_threshold:.0%}: {list(columns_to_drop)}")

    binary_vars = [col for col in data.columns if data[col].nunique() < 4]
    data[binary_vars] = data[binary_vars].fillna(0)

    continuous_vars = [col for col in data.columns if data[col].nunique() >= 10]
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    data[continuous_vars] = imputer.fit_transform(data[continuous_vars])

    categorical_vars = [col for col in data.columns if 4 <= data[col].nunique() < 10]
    if categorical_vars:
        encoder = OrdinalEncoder()
        data[categorical_vars] = encoder.fit_transform(data[categorical_vars])
        data[categorical_vars] = imputer.fit_transform(data[categorical_vars])
        data[categorical_vars] = encoder.inverse_transform(data[categorical_vars])
    return data


# ---------------------------
# Data Splitting, Sampling, and Feature Engineering
# ---------------------------
def split_data(all_data, label_col='HyperCoagulation'):
    """
    Split the data based on the 'source' column:
    - External data (source 0,1)
    - Internal data (source 3)
    Return X, y for each dataset.
    """
    out_data = all_data[all_data['source'].isin([0, 1])].drop('source', axis=1)
    in_data = all_data[all_data['source'] == 3].drop('source', axis=1)

    # Shuffle the internal data
    in_data = in_data.sample(frac=1, random_state=42).reset_index(drop=True)

    X_in = in_data.drop(label_col, axis=1)
    y_in = in_data[label_col]

    X_out = out_data.drop(label_col, axis=1)
    y_out = out_data[label_col]

    # Split internal data into training and internal test sets
    X_train, X_in_test, y_train, y_in_test = train_test_split(X_in, y_in, test_size=0.3,
                                                              random_state=42, stratify=y_in)
    return X_train, y_train, X_in_test, y_in_test, X_out, y_out


def perform_sampling(X_train, y_train, y_out):
    """
    Perform SMOTE sampling based on the positive rates of external and training data.
    """
    out_rate = y_out.sum() / len(y_out)
    train_rate = y_train.sum() / len(y_train)
    target_number = int(len(y_train) * out_rate)
    logging.info(f"SMOTE sampling: out_rate {out_rate:.2%}, train_rate {train_rate:.2%}, target_number {target_number}")

    if out_rate > train_rate:
        smote = SMOTE(sampling_strategy={1: target_number}, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    # Extend here for undersampling if needed
    return X_train, y_train


def select_features(X_train, y_train, X, top_n=50):
    """
    1. Select the top_n features based on Pearson correlation.
    2. Further selection (e.g., using LASSO) can be added (currently commented out).
    """
    correlations = X_train.corrwith(y_train)
    top_features = correlations.abs().nlargest(top_n).index
    selected_features = X[top_features].columns
    logging.info(f"Selected features: {list(selected_features)}")
    return selected_features


def standardize_data(X_train, X_in_test, X_out):
    """
    Standardize the data using StandardScaler and perform simple outlier treatment (clipping Z-scores).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_in_test_scaled = scaler.transform(X_in_test)
    X_out_scaled = scaler.transform(X_out)

    # Outlier treatment on training data
    z_scores = np.abs(stats.zscore(X_train_scaled))
    threshold = 3
    X_train_scaled[z_scores > threshold] = threshold
    return X_train_scaled, X_in_test_scaled, X_out_scaled, scaler


# ---------------------------
# Model Training and Evaluation
# ---------------------------
def get_models():
    """
    Define and return a dictionary of models.
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Neural Networks": MLPClassifier(max_iter=200),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": None  # XGBoost requires the xgboost package; set to None if not installed.
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier()
    except ImportError:
        logging.warning("xgboost not installed; skipping XGBoost.")
    return models


def parameter_optimization(models, param_grids, X_train, y_train):
    """
    Perform grid search for parameter optimization for each model.
    """
    best_models = {}
    for name, model in models.items():
        if name not in param_grids:
            best_models[name] = model
            continue
        logging.info(f"Optimizing {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        logging.info(f"Best parameters for {name}: {grid_search.best_params_}")
    return best_models


def evaluate_models(models, X_train, y_train, X_in_test, y_in_test, X_out, y_out,
                    selected_features):
    """
    Train and evaluate all models on training, internal test, and external test sets.
    Compute cross-validation metrics, plot ROC, Precision-Recall, calibration curves, etc.
    Save results and models.
    """
    # DataFrames for storing results
    train_95CI = pd.DataFrame()
    in_test_95CI = pd.DataFrame()
    out_test_95CI = pd.DataFrame()
    results = pd.DataFrame()
    bootstrapResults = pd.DataFrame()
    feature_importances = pd.DataFrame(index=selected_features)

    # Lists for storing curve data
    roc_data = []
    pr_curves = []
    calibration_data = []
    mcc_results_out = []
    mcc_results_in = []
    mcc_results_train = []

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
        'log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
        'mean_squared_error': make_scorer(lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2), needs_proba=True)
    }

    for name, model in models.items():
        logging.info(f"Training and evaluating model: {name}")
        model.fit(X_train, y_train)
        joblib.dump(model, f'./Death/Data/output/{name}.pkl')

        # Calculate metrics for train, internal test, and external test sets
        train_95CI_item = calculate_metrics(model, X_train, y_train, name)
        in_test_95CI_item = calculate_metrics(model, X_in_test, y_in_test, name)
        out_test_95CI_item = calculate_metrics(model, X_out, y_out, name)
        train_95CI = pd.concat([train_95CI, train_95CI_item])
        in_test_95CI = pd.concat([in_test_95CI, in_test_95CI_item])
        out_test_95CI = pd.concat([out_test_95CI, out_test_95CI_item])

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances[name] = model.coef_[0]

        # Cross-validation
        cv_scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        for key in cv_scores:
            if key.startswith('test_'):
                for i, score in enumerate(cv_scores[key]):
                    results.loc[f'{name}_{i}', key] = score

        # Bootstrap evaluation
        bootstrap_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for _ in range(100):
            X_resample, y_resample = resample(X_train, y_train)
            model.fit(X_resample, y_resample)
            y_pred = model.predict(X_train)
            bootstrap_scores['accuracy'].append(accuracy_score(y_train, y_pred))
            bootstrap_scores['precision'].append(precision_score(y_train, y_pred, average='weighted'))
            bootstrap_scores['recall'].append(recall_score(y_train, y_pred, average='weighted'))
            bootstrap_scores['f1'].append(f1_score(y_train, y_pred, average='weighted'))
        for metric in bootstrap_scores:
            bootstrapResults.loc[name, f'{metric} Mean'] = np.mean(bootstrap_scores[metric])
            bootstrapResults.loc[name, f'{metric} Lower 95% CI'] = np.percentile(bootstrap_scores[metric], 2.5)
            bootstrapResults.loc[name, f'{metric} Upper 95% CI'] = np.percentile(bootstrap_scores[metric], 97.5)

        # External test evaluation
        y_out_score = model.predict_proba(X_out)[:, 1]
        fpr, tpr, _ = roc_curve(y_out, y_out_score)
        roc_auc_val = auc(fpr, tpr)
        roc_data.append((fpr, tpr, roc_auc_val, name))

        # Precision-Recall curve and optimal threshold
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_out, y_out_score)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        preds = (y_out_score >= best_threshold).astype(int)
        pr_score = precision_score(y_out, preds)
        pr_curves.append((precision_vals, recall_vals, name, pr_score))

        # Calibration curve
        frac_pos, mean_pred = calibration_curve(y_out, y_out_score, n_bins=5)
        brier = brier_score_loss(y_out, y_out_score)
        calibration_data.append((frac_pos, mean_pred, name, brier))

        # MCC curves
        thresholds_out, mcc_scores_out = calculate_mcc_curve(y_out_score, X_out, y_out)
        mcc_results_out.append((thresholds_out, mcc_scores_out, name))
        y_in_score = model.predict_proba(X_in_test)[:, 1]
        thresholds_in, mcc_scores_in = calculate_mcc_curve(y_in_score, X_in_test, y_in_test)
        mcc_results_in.append((thresholds_in, mcc_scores_in, name))
        y_train_score = model.predict_proba(X_train)[:, 1]
        thresholds_train, mcc_scores_train = calculate_mcc_curve(y_train_score, X_train, y_train)
        mcc_results_train.append((thresholds_train, mcc_scores_train, name))
    # Save results
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results.to_csv(f'./Death/Data/output/cross_validation_results_{now_str}.csv')
    bootstrapResults.to_csv(f'./Death/Data/output/bootstrap_results_{now_str}.csv')
    train_95CI.to_csv(f'./Death/Data/output/train_95CI_results_{now_str}.csv')
    in_test_95CI.to_csv(f'./Death/Data/output/insidetest_95CI_results_{now_str}.csv')
    out_test_95CI.to_csv(f'./Death/Data/output/outside_95CI_results_{now_str}.csv')
    important_features = feature_importances[feature_importances > 0.0005].dropna(how='all')
    important_features.to_csv(f'./Death/Data/output/important_features_{now_str}.csv', encoding='UTF-8')

    # Plot ROC curve for external data
    plt.figure()
    for fpr, tpr, roc_auc_val, name in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f'{name} ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Outside ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'./Death/Data/output/Outside_roc_curves_{now_str}.png')
    plt.close()

    # Plot Precision-Recall and Calibration curves
    plot_precision_recall_curves(pr_curves, 'Precision-Recall Curve', f'./Death/Data/output/outside_precision_recall_curve_{now_str}.png')
    plot_calibration_curves(calibration_data, 'Calibration Curves', f'./Death/Data/output/outside_calibration_curves_{now_str}.png')
    plot_mcc_curves(mcc_results_out, f'./Death/Data/output/outside_mcc_curve_{now_str}.png')
    plot_mcc_curves(mcc_results_in, f'./Death/Data/output/insidetest_mcc_curve_{now_str}.png')
    plot_mcc_curves(mcc_results_train, f'./Death/Data/output/train_mcc_curve_{now_str}.png')


# ---------------------------
# Main Execution Flow
# ---------------------------
def main():
    # 1. Load data
    data_path = r"./Death/Data/output/filled_data1.csv"
    data = load_data(data_path)
    
    # Filter data with source values 0,1,2,3 and calculate positive rates
    filtered_data = data[data['source'].isin([0, 1, 2, 3])]
    positive_rates = calculate_positive_rate(filtered_data, [0, 1, 2, 3], 'HyperCoagulation')
    
    # Handle missing values and save processed data
    processed_data = handle_missing_values(filtered_data)
    processed_data.to_csv(r'./Death/Data/output/filled_data1_processed.csv', index=False, encoding='utf-8')
    logging.info("Processed data saved successfully.")

    # 2. Split data
    X_train, y_train, X_in_test, y_in_test, X_out, y_out = split_data(processed_data, label_col='HyperCoagulation')
    
    # 3. Sampling (SMOTE)
    X_train, y_train = perform_sampling(X_train, y_train, y_out)
    
    # 4. Feature Selection (based on Pearson correlation)
    selected_features = select_features(X_train, y_train, X_train)
    # Ensure that only selected features are used in subsequent datasets
    X_train = pd.DataFrame(X_train, columns=selected_features)
    X_in_test = pd.DataFrame(X_in_test, columns=selected_features)
    X_out = pd.DataFrame(X_out, columns=selected_features)

    # 5. Standardize data and treat outliers
    X_train_scaled, X_in_test_scaled, X_out_scaled, scaler = standardize_data(X_train, X_in_test, X_out)
    
    # 6. Define models and perform parameter optimization
    models = get_models()
    # Define parameter grid for each model
    param_grids = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        },
        "Decision Tree": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "K-Nearest Neighbors": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        "Neural Networks": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        "Naive Bayes": {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    models = parameter_optimization(models, param_grids, X_train_scaled, y_train)
    
    # 7. Train and evaluate models
    evaluate_models(models, X_train_scaled, y_train, X_in_test_scaled, y_in_test, X_out_scaled, y_out,
                    selected_features)
    
    # Optionally, use multi-threading to plot SHAP values (this can be time-consuming)
    # import concurrent.futures
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(plot_SHAP, name, model, X_train_scaled, selected_features) for name, model in models.items()]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             logging.error(f"Error plotting SHAP for a model: {e}")


if __name__ == '__main__':
    main()
