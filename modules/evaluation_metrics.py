from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def resolve_label_path(user_path: str) -> Path:
    """Resolve the label CSV path and support the common test_label.csv typo."""
    path = Path(user_path)
    if path.exists():
        return path

    if path.name == "test_label.csv":
        fallback = path.with_name("test_labels.csv")
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Label file not found: {path}")


def pick_label_columns(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    user_cols: list[str],
) -> list[str]:
    """Pick label columns from user input or the default toxic-comment label list."""
    if user_cols:
        cols = user_cols
    else:
        cols = [
            c
            for c in DEFAULT_LABEL_COLS
            if c in pred_df.columns and c in label_df.columns
        ]

    missing_pred = [c for c in cols if c not in pred_df.columns]
    missing_true = [c for c in cols if c not in label_df.columns]

    if missing_pred or missing_true:
        raise ValueError(
            f"Missing columns. In predictions: {missing_pred or 'None'}, "
            f"in labels: {missing_true or 'None'}"
        )

    if not cols:
        raise ValueError("No valid label columns found.")

    return cols


def align_rows(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align prediction and label rows by id when available; otherwise require equal row counts."""
    if "id" in pred_df.columns and "id" in label_df.columns:
        merged = label_df.merge(
            pred_df,
            on="id",
            how="inner",
            suffixes=("_true", "_pred"),
        )

        if merged.empty:
            raise ValueError("No matching ids between prediction and label files.")

        true_cols = [c for c in merged.columns if c.endswith("_true")]
        pred_cols = [c for c in merged.columns if c.endswith("_pred")]

        y_true = merged[["id"] + true_cols].copy()
        y_pred = merged[["id"] + pred_cols].copy()

        y_true.columns = ["id"] + [c.replace("_true", "") for c in true_cols]
        y_pred.columns = ["id"] + [c.replace("_pred", "") for c in pred_cols]

        return y_pred, y_true

    if len(pred_df) != len(label_df):
        raise ValueError(
            "Row mismatch and no id column for merge: "
            f"pred rows={len(pred_df)}, label rows={len(label_df)}"
        )

    return pred_df.copy(), label_df.copy()


def load_thresholds(path: Path, label_cols: list[str]) -> dict[str, float]:
    """Load per-label thresholds from a CSV with label names as index and a threshold column."""
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")

    thresholds_df = pd.read_csv(path, index_col=0)

    if "threshold" not in thresholds_df.columns:
        raise ValueError(f"Threshold file must contain a 'threshold' column: {path}")

    missing = [label for label in label_cols if label not in thresholds_df.index]
    if missing:
        raise ValueError(f"Threshold file is missing labels: {missing}")

    return {
        label: float(thresholds_df.loc[label, "threshold"])
        for label in label_cols
    }


def maybe_binarize_predictions(
    y_pred_raw: np.ndarray,
    threshold: float,
    per_label_thresholds: dict[str, float] | None,
    label_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert prediction scores to binary labels.
    """
    y_pred_raw = y_pred_raw.astype(float)
    is_binary = np.all(np.isin(y_pred_raw, [0.0, 1.0]))

    if is_binary:
        y_pred_bin = y_pred_raw.astype(int)
        threshold_info = {"threshold_mode": "already_binary"}
    elif per_label_thresholds is not None:
        thresholds_array = np.array(
            [per_label_thresholds[label] for label in label_cols],
            dtype=float,
        )
        y_pred_bin = (y_pred_raw >= thresholds_array).astype(int)
        threshold_info = {
            "threshold_mode": "per_label",
            "thresholds": per_label_thresholds,
        }
    else:
        y_pred_bin = (y_pred_raw >= threshold).astype(int)
        threshold_info = {
            "threshold_mode": "global",
            "threshold_used": threshold,
        }

    return y_pred_bin, y_pred_raw, threshold_info


def save_graphs(
    out_dir: Path,
    label_cols: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
) -> None:
    """Save metric graphs if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping graph generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(label_cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(label_cols, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-label Precision / Recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_label_prf.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(label_cols, support)
    ax.set_ylabel("Positive samples in ground truth")
    ax.set_title("Label Support")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "label_support.png", dpi=140)
    plt.close(fig)


def export_metrics(
    out_dir: Path,
    label_cols: list[str],
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    y_pred_scores: np.ndarray,
    threshold_info: dict,
) -> dict:
    """Calculate all metrics and export JSON/CSV/PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred_bin,
        average=None,
        zero_division=0,
    )

    subset_acc = float(accuracy_score(y_true, y_pred_bin))
    micro_f1 = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred_bin))

    per_label_auc = []
    valid_aucs = []
    for i in range(len(label_cols)):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_pred_scores[:, i]))
            per_label_auc.append(auc)
            valid_aucs.append(auc)
        except ValueError:
            per_label_auc.append(None)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    report = classification_report(
        y_true,
        y_pred_bin,
        target_names=label_cols,
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "num_total_rows_after_alignment": int(len(y_true)),
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss,
        "macro_roc_auc": macro_auc,
        "label_columns": label_cols,
    }
    summary.update(threshold_info)

    per_label_df = pd.DataFrame(
        {
            "label": label_cols,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": per_label_auc,
            "support": support.astype(int),
        }
    )

    metrics_json_path = out_dir / "metrics_summary.json"
    per_label_csv_path = out_dir / "per_label_metrics.csv"
    report_json_path = out_dir / "classification_report.json"

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_label_df.to_csv(per_label_csv_path, index=False)

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_graphs(out_dir, label_cols, precision, recall, f1, support)

    return {
        "summary": summary,
        "per_label_df": per_label_df,
        "classification_report": report,
        "metrics_json_path": metrics_json_path,
        "per_label_csv_path": per_label_csv_path,
        "report_json_path": report_json_path,
    }