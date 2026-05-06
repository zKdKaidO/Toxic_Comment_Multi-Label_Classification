import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class DeepLogisticRegression(nn.Module):
    """Single linear layer for multi-label classification on top of embeddings."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_sentence_embeddings(
    X_clean,
    X_test_clean,
    features_dir: str = "../features",
    model_name: str = "all-distilroberta-v1",
    show_progress_bar: bool = True,
):
    """
    Encode text lists with a SentenceTransformer model and save as .npy files.

    Parameters
    ----------
    X_clean        : list of preprocessed training texts
    X_test_clean   : list of preprocessed test texts
    features_dir   : directory to save .npy files
    model_name     : SentenceTransformer model name
    show_progress_bar : whether to show tqdm bars during encoding

    Returns
    -------
    X_train_embds, X_test_embds : np.ndarray
    """
    print(f"Loading SentenceTransformer model: '{model_name}'")
    model_st = SentenceTransformer(model_name)

    print("Encoding training data...")
    X_train_embds = model_st.encode(X_clean, show_progress_bar=show_progress_bar)
    print("Encoding test data...")
    X_test_embds = model_st.encode(X_test_clean, show_progress_bar=show_progress_bar)

    print(f"Training embeddings shape: {X_train_embds.shape}")
    print(f"Test embeddings shape:     {X_test_embds.shape}")

    os.makedirs(features_dir, exist_ok=True)
    np.save(f"{features_dir}/X_train_embeddings.npy", X_train_embds)
    np.save(f"{features_dir}/X_test_embeddings.npy", X_test_embds)
    print(f"Saved embeddings to '{features_dir}/'")

    return X_train_embds, X_test_embds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dl_model(
    X_train_dl: np.ndarray,
    y_train_dl: np.ndarray,
    models_dir: str = "../models",
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 0.01,
):
    """
    Train DeepLogisticRegression on pre-computed sentence embeddings.

    Uses smoothed class weights (sqrt(neg/pos)) for label imbalance.
    Saves model state dict to models_dir/pytorch_logreg_smoothed.pth.

    Returns
    -------
    dl_model : trained DeepLogisticRegression (on device)
    device   : torch.device used for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert X_train_dl.shape[0] == y_train_dl.shape[0], (
        "Feature and label row counts do not match!"
    )

    # Smoothed class weights for imbalanced labels
    num_pos = y_train_dl.sum(axis=0)
    num_neg = len(y_train_dl) - num_pos
    smoothed_weights = np.sqrt(num_neg / num_pos)
    pos_weight_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    # DataLoader
    X_tensor = torch.tensor(X_train_dl, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_dl, dtype=torch.float32)
    dataloader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    num_classes = y_train_dl.shape[1]
    dl_model = DeepLogisticRegression(X_train_dl.shape[1], num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=lr)

    print("\nSTARTING PYTORCH TRAINING...")
    start_time = time.time()

    for epoch in range(epochs):
        dl_model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(dl_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "pytorch_logreg_smoothed.pth")
    torch.save(dl_model.state_dict(), model_path)
    elapsed = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed:.2f} min. Saved to: {model_path}")

    return dl_model, device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dl_model(
    dl_model,
    device,
    X_test_dl: np.ndarray,
    test_df: pd.DataFrame,
    label_cols: list,
    truth_csv: str = "../test_labels.csv",
    models_dir: str = "../models",
    reports_dir: str = "../reports",
):
    """
    Run inference on test embeddings, merge with ground-truth labels,
    then compute and export metrics via evaluation_metrics.export_metrics.

    Outputs are saved to reports_dir/evaluation_output_dl/.

    Returns
    -------
    results dict from evaluation_metrics.export_metrics
    """
    from modules.evaluation_metrics import export_metrics, maybe_binarize_predictions

    # 1. Inference
    print("Generating predictions on the test set...")
    dl_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_dl, dtype=torch.float32).to(device)
        logits = dl_model(X_tensor)
        probs = torch.sigmoid(logits)
        y_test_proba_dl = probs.cpu().numpy()

    # 2. Save prediction probabilities
    test_ids = test_df["id"].values[: X_test_dl.shape[0]]
    dl_submission = pd.DataFrame(y_test_proba_dl, columns=label_cols)
    dl_submission.insert(0, "id", test_ids)

    os.makedirs(models_dir, exist_ok=True)
    dl_submission.to_csv(f"{models_dir}/test_pred_proba_dl.csv", index=False)
    print(f"Saved prediction probabilities to '{models_dir}/test_pred_proba_dl.csv'")

    # 3. Merge with ground truth and remove un-scored rows (-1)
    print("Evaluating comprehensive metrics...")
    truth_df = pd.read_csv(truth_csv)
    merged = pd.merge(truth_df, dl_submission, on="id", suffixes=("_true", "_pred"))
    clean_df = merged[merged["toxic_true"] != -1].copy()

    y_true = clean_df[[f"{c}_true" for c in label_cols]].values
    y_pred_probs = clean_df[[f"{c}_pred" for c in label_cols]].values

    # 4. Binarize and compute metrics
    threshold_info = {"threshold_mode": "global", "threshold_used": 0.5}
    y_pred_bin, y_pred_scores, _ = maybe_binarize_predictions(
        y_pred_probs,
        threshold=0.5,
        per_label_thresholds=None,
        label_cols=label_cols,
    )

    out_dir = Path(reports_dir) / "evaluation_output_dl"
    results = export_metrics(
        out_dir=out_dir,
        label_cols=label_cols,
        y_true=y_true,
        y_pred_bin=y_pred_bin,
        y_pred_scores=y_pred_scores,
        threshold_info=threshold_info,
    )

    # 5. Print summary
    summary = results["summary"]
    print("\n--- DEEP LEARNING EVALUATION COMPLETE ---")
    print(f"Rows evaluated:   {len(y_true)}")
    print(f"Subset accuracy:  {summary['subset_accuracy']:.4f}")
    print(f"Micro F1:         {summary['micro_f1']:.4f}")
    print(f"Macro F1:         {summary['macro_f1']:.4f}")
    print(f"Hamming loss:     {summary['hamming_loss']:.4f}")
    if summary["macro_roc_auc"] is not None:
        print(f"Macro ROC-AUC:    {summary['macro_roc_auc']:.4f}")
    print("\nPer-Label ROC-AUC:")
    for label, auc in zip(label_cols, results["per_label_df"]["roc_auc"]):
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {label:<15}: {auc_str}")
    print("-" * 40)

    from sklearn.metrics import classification_report as skl_cr
    print("\n" + skl_cr(y_true, y_pred_bin, target_names=label_cols, zero_division=0))

    print(f"\nExported metrics to '{out_dir}/':")
    print("  - metrics_summary.json")
    print("  - per_label_metrics.csv")
    print("  - classification_report.json")
    print("  - per_label_prf.png, label_support.png")

    return results
