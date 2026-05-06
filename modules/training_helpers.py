import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

RANDOM_STATE = 42
THRESHOLD_TUNE_SIZE = 0.20

def tune_thresholds(y_true_df, y_proba, label_cols, threshold_grid):
    """
    Tune one threshold per label using F1 score.
    This function must only be called on threshold-tuning data,
    not on the final evaluation fold.
    """
    best_thresholds = {}
    best_label_f1s = {}

    for idx, label in enumerate(label_cols):
        y_true_label = y_true_df[label].values
        best_t = 0.5
        best_f1 = -1.0

        for threshold in threshold_grid:
            y_pred_label = (y_proba[:, idx] >= threshold).astype(int)
            score = f1_score(y_true_label, y_pred_label, zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_t = float(threshold)

        best_thresholds[label] = best_t
        best_label_f1s[label] = best_f1

    return best_thresholds, best_label_f1s


def apply_thresholds(y_proba, thresholds, label_cols):
    """
    Convert probabilities to binary predictions using per-label thresholds.
    """
    return np.column_stack(
        [
            (y_proba[:, idx] >= thresholds[label]).astype(int)
            for idx, label in enumerate(label_cols)
        ]
    )


def compute_macro_metrics(y_true_df, y_pred):
    """
    Compute macro precision, recall, and F1.
    """
    macro_precision = precision_score(
        y_true_df, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true_df, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true_df, y_pred, average="macro", zero_division=0
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def split_fit_and_threshold_data(
    X,
    y_df,
    threshold_tune_size=THRESHOLD_TUNE_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Split data into:
    - model fitting data
    - threshold tuning data

    The threshold tuning data is used only to choose thresholds.
    """
    y_np = y_df.values

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=threshold_tune_size,
        random_state=random_state,
    )

    fit_idx, threshold_idx = next(
        splitter.split(np.zeros(len(y_np)), y_np)
    )

    X_fit = X[fit_idx]
    X_threshold = X[threshold_idx]

    y_fit = y_df.iloc[fit_idx].reset_index(drop=True)
    y_threshold = y_df.iloc[threshold_idx].reset_index(drop=True)

    return X_fit, X_threshold, y_fit, y_threshold