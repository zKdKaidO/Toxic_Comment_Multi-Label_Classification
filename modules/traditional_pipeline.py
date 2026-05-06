import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import h5sparse
import joblib
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def fit_tfidf(
    X_normalize,
    X_test_normalize,
    max_features: int = 10000,
    features_dir: str = "../features",
):
    """
    Fit TF-IDF on normalised training text, transform test text,
    and save both sparse matrices to .h5 files.

    Returns
    -------
    X_train_tfidf, X_test_tfidf, tfidf : sparse matrices + fitted vectoriser
    """
    print("Fitting TF-IDF on training data...")
    tfidf = TfidfVectorizer(min_df=1, max_features=max_features)
    X_train_tfidf = tfidf.fit_transform(X_normalize)
    print(f"  X_train shape: {X_train_tfidf.shape}")

    print("Transforming test data...")
    X_test_tfidf = tfidf.transform(X_test_normalize)
    print(f"  X_test shape:  {X_test_tfidf.shape}")

    os.makedirs(features_dir, exist_ok=True)
    with h5sparse.File(f"{features_dir}/tfidf_train_embeddings.h5", "w") as h5f:
        h5f.create_dataset("X_train_tfidf", data=X_train_tfidf)
    with h5sparse.File(f"{features_dir}/tfidf_test_embeddings.h5", "w") as h5f:
        h5f.create_dataset("X_test_tfidf", data=X_test_tfidf)

    print(f"Saved TF-IDF embeddings to '{features_dir}/'")
    return X_train_tfidf, X_test_tfidf, tfidf


# ---------------------------------------------------------------------------
# End-to-End Training (Optuna + threshold tuning)
# ---------------------------------------------------------------------------

def run_optuna_training(
    X_train_tfidf,
    X_test_tfidf,
    y_all,
    test_df: pd.DataFrame,
    label_cols: list,
    models_dir: str = "../models",
    n_trials: int = 20,
):
    """
    Full Traditional Pipeline:
      1. Multilabel-stratified split (80 fit / 20 temp)
      2. Inner split of temp (50 tune / 50 report)
      3. Optuna hyperparameter search (3-fold CV on fit split)
      4. Train final model on fit split
      5. Per-label threshold tuning on tune split
      6. Evaluate on report split
      7. Retrain on 100 % of training data
      8. Save test predictions + thresholds

    Returns
    -------
    final_model, best_thresholds
    """
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_all_np = y_all.values
    threshold_grid = np.arange(0.1, 0.95, 0.005)

    # 1. Outer split
    msss_outer = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    fit_idx, temp_idx = next(
        msss_outer.split(np.zeros(len(y_all_np)), y_all_np)
    )
    X_fit = X_train_tfidf[fit_idx]
    y_fit = y_all.iloc[fit_idx].reset_index(drop=True)
    X_temp = X_train_tfidf[temp_idx]
    y_temp = y_all.iloc[temp_idx].reset_index(drop=True)

    # 2. Inner split
    msss_inner = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=42
    )
    tune_idx, report_idx = next(
        msss_inner.split(np.zeros(len(y_temp)), y_temp.values)
    )
    X_val_tune  = X_temp[tune_idx]
    y_val_tune  = y_temp.iloc[tune_idx].reset_index(drop=True)
    X_val_report = X_temp[report_idx]
    y_val_report = y_temp.iloc[report_idx].reset_index(drop=True)

    # 3. Optuna objective
    def objective(trial):
        c_val       = trial.suggest_float("C", 1e-3, 10.0, log=True)
        penalty_val = trial.suggest_categorical("penalty", ["l2"])
        solver_val  = trial.suggest_categorical("solver", ["lbfgs"])

        base = LogisticRegression(
            C=c_val, penalty=penalty_val, solver=solver_val,
            class_weight="balanced", max_iter=1500,
            random_state=42, n_jobs=-1,
        )
        mskf = MultilabelStratifiedKFold(
            n_splits=3, shuffle=True, random_state=42
        )
        fold_scores = []
        for tr_idx, va_idx in mskf.split(np.zeros(len(y_fit)), y_fit.values):
            clf = OneVsRestClassifier(base)
            clf.fit(X_fit[tr_idx], y_fit.iloc[tr_idx])
            y_va_proba = clf.predict_proba(X_fit[va_idx])
            y_va = y_fit.iloc[va_idx]
            label_f1s = []
            for idx, label in enumerate(label_cols):
                best_f1 = max(
                    f1_score(
                        y_va[label].values,
                        (y_va_proba[:, idx] >= t).astype(int),
                        zero_division=0,
                    )
                    for t in threshold_grid
                )
                label_f1s.append(best_f1)
            fold_scores.append(np.mean(label_f1s))
        return float(np.mean(fold_scores))

    # 4. Run Optuna
    print("Starting Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Optimization complete! Best params: {study.best_params}")

    # 5. Train final model on fit split
    bp = study.best_params
    best_base = LogisticRegression(
        C=bp["C"], penalty=bp["penalty"], solver=bp["solver"],
        class_weight="balanced", max_iter=1500, random_state=42, n_jobs=-1,
    )
    final_model = OneVsRestClassifier(best_base)
    print("Training final model with best parameters...")
    final_model.fit(X_fit, y_fit)
    print("Final model trained successfully!")

    # 6. Per-label threshold tuning
    print("Tuning per-label thresholds on threshold-tuning split...")
    y_tune_proba = final_model.predict_proba(X_val_tune)
    best_thresholds = {}
    for idx, label in enumerate(label_cols):
        best_t, best_f1 = 0.5, -1.0
        for t in threshold_grid:
            score = f1_score(
                y_val_tune[label].values,
                (y_tune_proba[:, idx] >= t).astype(int),
                zero_division=0,
            )
            if score > best_f1:
                best_f1, best_t = score, float(t)
        best_thresholds[label] = best_t
    print(f"Best per-label thresholds: {best_thresholds}")

    # 7. Evaluate on report split
    print("\nEvaluating on holdout report split...")
    y_report_proba = final_model.predict_proba(X_val_report)
    y_report_pred = np.column_stack([
        (y_report_proba[:, idx] >= best_thresholds[label]).astype(int)
        for idx, label in enumerate(label_cols)
    ])
    print(classification_report(y_val_report, y_report_pred, target_names=label_cols))
    print(f"Macro Precision: {precision_score(y_val_report, y_report_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall:    {recall_score(y_val_report, y_report_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro F1:        {f1_score(y_val_report, y_report_pred, average='macro', zero_division=0):.4f}")

    # 8. Retrain on 100 % of training data
    print("\nRetraining final model on 100 % of training data...")
    final_model.fit(X_train_tfidf, y_all)

    # 9. Save test outputs
    print("Generating predictions for test TF-IDF features...")
    y_test_proba = final_model.predict_proba(X_test_tfidf)
    y_test_pred = np.column_stack([
        (y_test_proba[:, idx] >= best_thresholds[label]).astype(int)
        for idx, label in enumerate(label_cols)
    ])

    os.makedirs(models_dir, exist_ok=True)
    pd.Series(best_thresholds, name="threshold").to_csv(
        f"{models_dir}/best_thresholds.csv"
    )

    test_proba_df = pd.DataFrame(y_test_proba, columns=label_cols)
    if "id" in test_df.columns:
        test_proba_df.insert(0, "id", test_df["id"].values)
    test_proba_df.to_csv(f"{models_dir}/test_pred_proba.csv", index=False)

    test_binary_df = pd.DataFrame(y_test_pred, columns=label_cols)
    if "id" in test_df.columns:
        test_binary_df.insert(0, "id", test_df["id"].values)
    test_binary_df.to_csv(f"{models_dir}/test_pred_binary.csv", index=False)

    print(f"Saved thresholds, probabilities, and binary predictions to '{models_dir}/'")
    return final_model, best_thresholds


# ---------------------------------------------------------------------------
# Simple helper (kept for backward-compatibility)
# ---------------------------------------------------------------------------

def train_traditional_model(
    train_df: pd.DataFrame,
    text_column: str,
    label_columns: list,
    model_save_path: str,
):
    """
    Huấn luyện mô hình TF-IDF + Logistic Regression và lưu lại.
    """
    print("Bắt đầu huấn luyện pipeline truyền thống...")

    print("Tạo TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df[text_column])
    y_train = train_df[label_columns].values

    print("Huấn luyện OneVsRestClassifier với LogisticRegression...")
    logreg = LogisticRegression(solver="liblinear", random_state=42)
    classifier = OneVsRestClassifier(logreg)
    classifier.fit(X_train, y_train)

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_save_path)
    print(f"Đã lưu model và vectorizer tại: {model_save_path}")

    return classifier, vectorizer