"""
Assumptions
  • Your CSV has a binary label column named CFG.label_col
  • All feature columns are numeric or can be coerced; basic handling for
    categoricals is sketched (commented) below if you need it.

Usage
  1) Set CFG.data_path to your CSV path.
  2) Run the script to train; it saves `threat_detector.keras`.
  3) Use detect_threat(...) for inference with the chosen threshold.

Notes
  • For real deployments, wrap this with CLI/argparse or a web service.
  • Add schema validation (pydantic) in production to catch malformed inputs.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)

# ============================
# Reproducibility controls
# ============================
# Setting seeds improves repeatability. GPU ops can still be nondeterministic
# on some kernels/drivers, but this helps keep runs comparable.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================
# Paths (tailored to the folder layout)
# ============================
# This file lives at project/scripts/threat_detection.py
# ROOT is the project root (one level up from scripts/)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================
# Configuration block
# ============================
# Keep common knobs in one place so experiments are easy to reproduce.
@dataclass
class Config:
    data_path: str = str(DATA_DIR / "threats_dataset.csv")  # <-- change me
    label_col: str = "is_threat"              # binary label column (0/1)
    test_size: float = 0.2                     # fraction for final test split
    val_size: float = 0.2                      # fraction of TRAIN used for val
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10                         # early stopping patience (epochs)
    learning_rate: float = 1e-3
    hidden_units: Tuple[int, ...] = (128, 64, 32)  # MLP architecture
    dropout: float = 0.3
    # Threshold strategy used on the validation set to pick the decision cutoff
    threshold_metric: str = "f1"  # options: "f1", "youden", "recall@precision>=0.9"

CFG = Config()


# ============================
# Data loading & basic cleanup
# ============================
# Reads a CSV, splits features/label, and performs very light cleaning.
# For real pipelines, add:
#   • dedicated missing‑value strategy (per column)
#   • categorical encoders (OneHot/Embeddings)
#   • type assertions / schema validation

def load_dataframe(path: str, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return (X, y).

    Parameters
    ----------
    path : str
        Path to CSV file.
    label_col : str
        Name of the binary label column (0/1).
    """
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}.")

    # y is the label (ensure integer type 0/1)
    y = df[label_col].astype(int)

    # X is everything else
    X = df.drop(columns=[label_col]).copy()

    # ---- Minimal numeric/categorical handling ----
    # If a column is numeric: cast to float and median‑impute NaNs.
    # If a column is non‑numeric: here we coerce to category codes as a simple
    # fallback. For production, prefer OneHotEncoder or embeddings.
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].astype(float)
            X[col] = X[col].fillna(X[col].median())
        else:
            # NOTE: Replace this with a proper encoder when you have real categoricals.
            X[col] = X[col].astype("category").cat.codes
            # Pandas uses -1 for unknown; map to 0 to avoid negatives.
            X[col] = X[col].replace(-1, np.nan).fillna(0)

    return X, y


# ============================
# Train/Val/Test splits
# ============================
# We perform a stratified split to preserve label ratios across splits.

def make_splits(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float):
    """Return ((X_tr,y_tr),(X_val,y_val),(X_te,y_te))."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=SEED
    )
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)


# ============================
# tf.data helpers
# ============================
# Using tf.data gives you streaming, shuffling, and prefetching for performance.

def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y.astype("float32")))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================
# Model definition
# ============================
# The Normalization layer is part of the model so the saved artifact
# encapsulates preprocessing (no external scaler object needed).

def build_model(n_features: int, hidden: Tuple[int, ...], dropout: float, lr: float):
    """Create a small MLP with in‑graph Normalization.

    Returns
    -------
    model : tf.keras.Model
        Compiled model ready to train.
    norm : tf.keras.layers.Normalization
        The Normalization layer (so we can call .adapt on train data only).
    """
    inputs = tf.keras.Input(shape=(n_features,), name="features")

    # In‑graph standardization: learns mean/variance on TRAIN only via adapt().
    norm = tf.keras.layers.Normalization(axis=-1, name="norm")
    x = norm(inputs)

    # Feed‑forward backbone
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # Sigmoid output for binary probability
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="proba")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(curve="ROC", name="auc"),   # ROC‑AUC
            tf.keras.metrics.AUC(curve="PR", name="auprc"),  # PR‑AUC
        ],
    )
    return model, norm


# ============================
# Class weights (imbalance handling)
# ============================
# If positives are rare, this increases their contribution to the loss.

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    # w_class = total/(2*count_class) — balances the sum of weights by class
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    total = n_pos + n_neg
    # Avoid division by zero for degenerate datasets
    n_pos = max(1, n_pos)
    n_neg = max(1, n_neg)
    return {0: total / (2.0 * n_neg), 1: total / (2.0 * n_pos)}


# ============================
# Threshold tuning strategies
# ============================
# We choose the classification threshold on the VALIDATION set, not on test.

def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "f1") -> float:
    """
    Pick a decision threshold according to a strategy.

    Supported strategies:
      • "f1" – maximizes F1 on the validation set
      • "youden" – maximizes Youden's J (TPR − FPR) over a grid
      • "recall@precision>=X" – pick threshold that maximizes recall subject to
        precision ≥ X (e.g., "recall@precision>=0.9")
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # sklearn returns len(thresholds) = len(precision) - 1; pad for alignment
    thresholds = np.append(thresholds, 1.0)

    if strategy == "f1":
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = np.nanargmax(f1)
        return float(thresholds[idx])

    elif strategy == "youden":
        # Scan a fixed grid; simple and robust.
        grid = np.linspace(0.0, 1.0, 501)
        best, best_t = -1.0, 0.5
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            j = tpr - fpr
            if j > best:
                best, best_t = j, t
        return float(best_t)

    else:
        # Parse patterns like "recall@precision>=0.9"
        if strategy.startswith("recall@precision>="):
            target_p = float(strategy.split(">=")[-1])
            feasible = np.where(precision >= target_p)[0]
            if feasible.size > 0:
                # Among feasible points, pick the highest recall
                idx = feasible[np.argmax(recall[feasible])]
                return float(thresholds[idx])
        # Fallback to the common 0.5 if strategy unrecognized
        return 0.5


# ============================
# Training & evaluation
# ============================
# Trains the model with early stopping, chooses a threshold on the val set,
# reports metrics on the test set at that threshold, and saves the model.

def train_and_evaluate(cfg: Config) -> Dict[str, Any]:
    # --- Load and split data ---
    X_df, y_s = load_dataframe(cfg.data_path, cfg.label_col)
    (X_tr_df, y_tr_s), (X_val_df, y_val_s), (X_te_df, y_te_s) = make_splits(
        X_df, y_s, cfg.test_size, cfg.val_size
    )

    # Convert to numpy for tf.data
    X_tr = X_tr_df.values.astype("float32")
    X_val = X_val_df.values.astype("float32")
    X_te = X_te_df.values.astype("float32")
    y_tr = y_tr_s.values.astype("float32")
    y_val = y_val_s.values.astype("float32")
    y_te = y_te_s.values.astype("float32")

    # --- Build model ---
    n_features = X_tr.shape[1]
    model, norm = build_model(n_features, cfg.hidden_units, cfg.dropout, cfg.learning_rate)

    # IMPORTANT: fit normalization ONLY on training data to avoid leakage
    norm.adapt(X_tr, batch_size=min(len(X_tr), 1024))

    # Build efficient datasets
    train_ds = make_tf_dataset(X_tr, y_tr, cfg.batch_size, training=True)
    val_ds = make_tf_dataset(X_val, y_val, cfg.batch_size, training=False)
    test_ds = make_tf_dataset(X_te, y_te, cfg.batch_size, training=False)

    # Callbacks: stop when val AUC stops improving; keep the best model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=cfg.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    # Class weights help with imbalance (rare threats)
    class_weight = compute_class_weights(y_tr)

    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # --- Threshold selection on validation set ---
    y_val_prob = model.predict(val_ds, verbose=0).ravel()
    thr = pick_threshold(y_val, y_val_prob, strategy=cfg.threshold_metric)

    # --- Final evaluation on test set using the chosen threshold ---
    y_te_prob = model.predict(test_ds, verbose=0).ravel()
    y_te_pred = (y_te_prob >= thr).astype(int)

    print(f"Chosen threshold (on val): {thr:.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_te, y_te_prob):.4f}")
    print(f"Test PR-AUC:  {average_precision_score(y_te, y_te_prob):.4f}")
    print(confusion_matrix(y_te, y_te_pred))
    print(classification_report(y_te, y_te_pred, digits=4))

    # Save the entire model (architecture + weights + in‑graph preprocessing)
    model.save("threat_detector.keras")

    return {
        "model": model,
        "threshold": float(thr),
        "n_features": int(n_features),
        "history": history.history,
    }


# ============================
# Inference utilities
# ============================
# Use these in a service or notebook to score new samples.

def detect_threat(model: tf.keras.Model, input_data: np.ndarray, threshold: float):
    """Predict threat labels and probabilities.

    Parameters
    ----------
    model : tf.keras.Model
        A trained model saved by this script (includes Normalization).
    input_data : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix; same column order as during training.
    threshold : float
        Decision cutoff chosen on the validation set.

    Returns
    -------
    is_threat : np.ndarray of bool, shape (n_samples,)
        True where predicted probability ≥ threshold.
    confidence : np.ndarray of float, shape (n_samples,)
        Predicted probabilities in [0, 1].
    """
    # Ensure proper dtype and shape
    probs = model.predict(input_data.astype("float32"), verbose=0).ravel()
    preds = probs >= float(threshold)
    return preds.astype(bool), probs


# ============================
# Script entry point
# ============================
if __name__ == "__main__":
    cfg = CFG

    # Train and evaluate; artifacts contain the trained model and chosen threshold
    artifacts = train_and_evaluate(cfg)

    # -------- Example inference --------
    # Replace the '...' with real numbers of length artifacts["n_features"].
    # new_sample = np.array([[...]], dtype="float32")
    # is_threat, confidence = detect_threat(
    #     artifacts["model"], new_sample, artifacts["threshold"]
    # )
    # print(f"Threat detected: {bool(is_threat[0])}, Confidence: {confidence[0]:.2f}")
