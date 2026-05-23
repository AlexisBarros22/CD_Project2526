"""
model_training.py
Comprehensive model training for flight delay prediction.

Classes
-------
KNNScratch          - k-NN from scratch (numpy only), regression & classification
SupervisedModels    - DecisionTreeClassifier + DecisionTreeRegressor
EnsembleModels      - Random Forest (bagging) + LightGBM (boosting)
DeepLearningModel   - PyTorch MLP, regression & classification
ClusteringModels    - KMeans (k=2-6) + DBSCAN
ModelComparison     - unified comparison table and plots
ModelTraining       - kept for backward compatibility (LightGBM regression)

Helpers
-------
winsorize_target    - cap regression target at a percentile to suppress outliers

Classification target (3 classes)
----------------------------------
0 - on-time     : ARR_DELAY < 15 min
1 - short delay : 15 <= ARR_DELAY <= 30 min
2 - long delay  : ARR_DELAY > 30 min
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score, davies_bouldin_score,
)

# LightGBM
from lightgbm import LGBMRegressor, LGBMClassifier

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

CLASS_LABELS = ["On-time (<15 min)", "Short delay (15-30 min)", "Long delay (>30 min)"]


def make_class_labels(y: np.ndarray, t1: int = 15, t2: int = 30) -> np.ndarray:
    """Convert continuous ARR_DELAY values to 3-class integer labels."""
    y = np.asarray(y, dtype=float)
    out = np.zeros(len(y), dtype=int)
    out[y >= t1] = 1
    out[y > t2] = 2
    return out


def winsorize_target(y: np.ndarray, percentile: float = 99) -> np.ndarray:
    """Cap y at the given upper percentile to suppress outlier influence on regression."""
    cap = float(np.percentile(np.asarray(y, dtype=float), percentile))
    return np.minimum(np.asarray(y, dtype=float), cap)


def _ensure_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reg_metrics(y_true, y_pred, prefix: str = "") -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": np.sqrt(mse),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
    }


def _cls_metrics(y_true, y_pred, prefix: str = "") -> dict:
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _plot_confusion(y_true, y_pred, title: str, save_path: Path | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
    labels = ["On-time\n(<15 min)", "Short\n(15-30 min)", "Long\n(>30 min)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[0],
        linewidths=0.5, linecolor="white", annot_kws={"size": 12},
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)
    axes[0].set_title("Counts", fontsize=12, fontweight="bold")

    annot_pct = np.array([[f"{v:.0%}" for v in row] for row in cm_norm])
    sns.heatmap(
        cm_norm, annot=annot_pct, fmt="", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels, ax=axes[1],
        vmin=0, vmax=1, linewidths=0.5, linecolor="white", annot_kws={"size": 12},
    )
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)
    axes[1].set_title("Row-normalised (%)", fontsize=12, fontweight="bold")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    fig.suptitle(
        f"{title}\nAccuracy: {acc:.3f}  |  F1 (weighted): {f1:.3f}",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _plot_residuals(y_true, y_pred, title: str, save_path: Path | None = None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    r2 = r2_score(y_true, y_pred)

    N = len(y_true)
    if N > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, 50_000, replace=False)
    else:
        idx = np.arange(N)
    yt, yp, res = y_true[idx], y_pred[idx], residuals[idx]
    abs_res = np.abs(res)
    clim = float(np.percentile(abs_res, 95))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Actual vs Predicted
    sc = axes[0].scatter(
        yp, yt, c=abs_res, cmap="YlOrRd", s=8, alpha=0.4,
        vmin=0, vmax=clim,
    )
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    axes[0].plot([lo, hi], [lo, hi], color="#2166ac", lw=1.8, ls="--", label="Perfect fit")
    axes[0].set_xlabel("Predicted (min)", fontsize=11)
    axes[0].set_ylabel("Actual (min)", fontsize=11)
    axes[0].set_title("Actual vs Predicted", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    plt.colorbar(sc, ax=axes[0], label="|Residual| (min)")

    # Panel 2: Residuals vs Predicted
    norm2 = plt.Normalize(-clim, clim)
    axes[1].scatter(yp, res, c=res, cmap="coolwarm", s=8, alpha=0.4, norm=norm2)
    axes[1].axhline(0, color="#d62728", lw=1.8, ls="--")
    axes[1].set_xlabel("Predicted (min)", fontsize=11)
    axes[1].set_ylabel("Residual (min)", fontsize=11)
    axes[1].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")

    # Panel 3: Residual distribution
    axes[2].hist(residuals, bins=80, color="#4393c3", edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color="#d62728", lw=1.8, ls="--")
    axes[2].set_xlabel("Residual (min)", fontsize=11)
    axes[2].set_ylabel("Count", fontsize=11)
    axes[2].set_title("Residual Distribution", fontsize=12, fontweight="bold")
    axes[2].text(
        0.97, 0.97,
        f"RMSE: {rmse:.2f}\nMAE:  {mae:.2f}\nR2:   {r2:.4f}",
        transform=axes[2].transAxes, fontsize=10, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffde7", alpha=0.85),
    )

    fig.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 1. kNN from scratch (numpy only)
# ──────────────────────────────────────────────────────────────────────────────

class KNNScratch:
    """
    k-Nearest Neighbours implemented from scratch using only numpy.

    Both regression (predict mean of k neighbours) and classification
    (predict majority class among k neighbours) are supported.

    Because the full dataset (~2.3 M rows) would make neighbour search
    prohibitively slow, the class subsamples both training and test sets
    for demonstration.  The subsample sizes are configurable.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    task : {'classification', 'regression'}
        Whether to predict class labels or continuous values.
    n_train_samples : int
        Number of training rows to keep (random subsample).
    n_test_samples : int
        Number of test rows to evaluate.
    random_state : int
        Seed for reproducible subsampling.
    class_weight : {'balanced', None}
        If 'balanced', votes are weighted by inverse class frequency, matching
        the class_weight='balanced' setting used by the other classifiers.
        Only used when task='classification'.
    verbose : bool
    """

    def __init__(
        self,
        k: int = 5,
        task: str = "classification",
        n_train_samples: int = 20_000,
        n_test_samples: int = 5_000,
        random_state: int = 48,
        class_weight: str | None = "balanced",
        verbose: bool = True,
    ):
        self.k = k
        self.task = task
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.random_state = random_state
        self.class_weight = class_weight
        self.verbose = verbose

        self._X_tr: np.ndarray | None = None
        self._y_tr: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._class_weights: np.ndarray | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    def _subsample(self, X, y, n: int, seed: int | None = None):
        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return np.asarray(X, dtype=float)[idx], np.asarray(y, dtype=float)[idx]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization fitted on training data."""
        return (X - self._mean) / self._std

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train) -> "KNNScratch":
        """Store a subsampled, z-score-normalised copy of the training set."""
        X, y = self._subsample(X_train, y_train, self.n_train_samples)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0  # avoid division by zero on constant cols

        self._X_tr = self._normalize(X)
        self._y_tr = make_class_labels(y) if self.task == "classification" else y

        if self.task == "classification" and self.class_weight == "balanced":
            classes, counts = np.unique(self._y_tr, return_counts=True)
            n_samples = len(self._y_tr)
            n_classes = len(classes)
            self._class_weights = np.ones(3)
            for cls, cnt in zip(classes.astype(int), counts):
                self._class_weights[cls] = n_samples / (n_classes * cnt)

        if self.verbose:
            weight_str = f", class_weight={self.class_weight}" if self.task == "classification" else ""
            print(
                f"[KNN] Training set: {len(self._X_tr):,} samples | "
                f"k={self.k} | task={self.task}{weight_str}"
            )
        return self

    # ------------------------------------------------------------------
    def _predict_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Predict for a small batch of normalised test samples.

        Uses the algebraic identity
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
        for vectorised distance computation.
        """
        a2 = (X_batch ** 2).sum(axis=1, keepdims=True)          # (b, 1)
        b2 = (self._X_tr ** 2).sum(axis=1, keepdims=True).T     # (1, n)
        ab = X_batch @ self._X_tr.T                              # (b, n)
        dists = np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0.0))      # (b, n)

        nn_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        nn_y = self._y_tr[nn_idx]                                # (b, k)

        if self.task == "classification":
            nn_y_int = nn_y.astype(int)
            if self._class_weights is not None:
                votes = np.stack(
                    [self._class_weights[c] * (nn_y_int == c).sum(axis=1) for c in range(3)],
                    axis=1,
                )
                preds = votes.argmax(axis=1)
            else:
                preds = np.array(
                    [np.bincount(row, minlength=3).argmax() for row in nn_y_int]
                )
        else:
            preds = nn_y.mean(axis=1)

        return preds

    # ------------------------------------------------------------------
    def predict(self, X_test, batch_size: int = 500) -> np.ndarray:
        """Predict on the full (normalised) test array in batches."""
        X = self._normalize(np.asarray(X_test, dtype=float))
        n = len(X)
        out_dtype = int if self.task == "classification" else float
        preds = np.empty(n, dtype=out_dtype)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            preds[start:end] = self._predict_batch(X[start:end])

        return preds

    # ------------------------------------------------------------------
    def evaluate(self, y_true, y_pred) -> dict:
        if self.task == "classification":
            y_cls = make_class_labels(y_true)
            self.metrics = _cls_metrics(y_cls, y_pred)
        else:
            self.metrics = _reg_metrics(y_true, y_pred)
        return self.metrics

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir: str | None = None,
    ) -> dict:
        """
        Full pipeline: fit → subsample test → predict → evaluate → (plot).

        Returns
        -------
        dict  metrics dictionary
        """
        out = _ensure_dir(output_dir)
        self.fit(X_train, y_train)

        X_ts, y_ts = self._subsample(X_test, y_test, self.n_test_samples, seed=99)
        if self.verbose:
            print(f"[KNN] Predicting on {len(X_ts):,} test samples ...")

        y_pred = self.predict(X_ts)
        metrics = self.evaluate(y_ts, y_pred)

        if self.verbose:
            tag = f"KNN (k={self.k}, {self.task})"
            print(f"\n{'=' * 52}\n{tag} RESULTS\n{'=' * 52}")
            for name, val in metrics.items():
                print(f"  {name:<22}: {val:.4f}")
            print()

        if out is not None:
            fname = f"knn_{self.task}_confusion.png" if self.task == "classification" else f"knn_{self.task}_residuals.png"
            if self.task == "classification":
                _plot_confusion(
                    make_class_labels(y_ts), y_pred,
                    f"KNN k={self.k} - Confusion Matrix",
                    save_path=out / fname,
                )
            else:
                _plot_residuals(
                    y_ts, y_pred,
                    f"KNN k={self.k} - Regression Residuals",
                    save_path=out / fname,
                )
            if self.verbose:
                print(f"[KNN] Plot saved to {out / fname}")

        # Add test_ prefix so keys match ModelComparison expectations
        return {f"test_{k}": v for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Supervised models (sklearn)
# ──────────────────────────────────────────────────────────────────────────────

class SupervisedModels:
    """
    Classical supervised learning models via scikit-learn.

    Regression model
    ----------------
    - DecisionTreeRegressor (max_depth=12)

    Classification model
    --------------------
    - DecisionTreeClassifier (max_depth=12)

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like  (continuous ARR_DELAY values, used for classification labels)
    y_train_reg, y_test_reg : array-like, optional
        Winsorized regression targets. If None, falls back to y_train / y_test.
        Classification labels are always derived from the original y_train / y_test.
    verbose : bool
    """

    def __init__(self, X_train, X_test, y_train, y_test,
                 y_train_reg=None, y_test_reg=None, verbose: bool = True):
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        self.X_train = np.asarray(X_train, dtype=float)
        self.X_test = np.asarray(X_test, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float)
        self.y_test = np.asarray(y_test, dtype=float)
        self.y_train_cls = make_class_labels(self.y_train)
        self.y_test_cls = make_class_labels(self.y_test)
        self.y_train_reg = np.asarray(y_train_reg if y_train_reg is not None else y_train, dtype=float)
        self.y_test_reg = np.asarray(y_test_reg if y_test_reg is not None else y_test, dtype=float)
        self.verbose = verbose

        self._reg_results: dict[str, dict] = {}
        self._cls_results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def _fit_eval_reg(self, name: str, model) -> dict:
        if self.verbose:
            print(f"  Training {name} ...")
        model.fit(self.X_train, self.y_train_reg)
        tr_pred = model.predict(self.X_train)
        te_pred = model.predict(self.X_test)
        m = {
            **_reg_metrics(self.y_train_reg, tr_pred, "train_"),
            **_reg_metrics(self.y_test_reg, te_pred, "test_"),
            "_model": model,
            "_te_pred": te_pred,
        }
        self._reg_results[name] = m
        return m

    def _fit_eval_cls(self, name: str, model) -> dict:
        if self.verbose:
            print(f"  Training {name} ...")
        model.fit(self.X_train, self.y_train_cls)
        tr_pred = model.predict(self.X_train)
        te_pred = model.predict(self.X_test)
        m = {
            **_cls_metrics(self.y_train_cls, tr_pred, "train_"),
            **_cls_metrics(self.y_test_cls, te_pred, "test_"),
            "_model": model,
            "_te_pred": te_pred,
        }
        self._cls_results[name] = m
        return m

    # ------------------------------------------------------------------
    def train_all(self) -> "SupervisedModels":
        # ── Classification first ──
        if self.verbose:
            print("\n[Supervised] Training classification models ...")
        self._fit_eval_cls(
            "Decision Tree (cls)",
            DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=48),
        )

        # ── Regression second ──
        if self.verbose:
            print("[Supervised] Training regression models ...")
        self._fit_eval_reg(
            "Decision Tree (reg)",
            DecisionTreeRegressor(max_depth=12, random_state=48),
        )
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _model_dir(base: Path | None, model_name: str) -> Path | None:
        if base is None:
            return None
        d = base / model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plot_results(self, output_dir: str | None = None) -> "SupervisedModels":
        out = _ensure_dir(output_dir)

        for name, m in self._reg_results.items():
            model_out = self._model_dir(out, name)
            path = (model_out / "regression_residuals.png") if model_out else None
            _plot_residuals(self.y_test_reg, m["_te_pred"], f"{name} - Regression Results", path)

        for name, m in self._cls_results.items():
            model_out = self._model_dir(out, name)
            path = (model_out / "classification_confusion.png") if model_out else None
            _plot_confusion(self.y_test_cls, m["_te_pred"], f"{name} - Confusion Matrix", path)

        if out and self.verbose:
            print(f"[Supervised] Plots saved to {out}")
        return self

    # ------------------------------------------------------------------
    def print_summary(self) -> "SupervisedModels":
        print("\n" + "=" * 60)
        print("SUPERVISED MODELS - REGRESSION RESULTS")
        print("=" * 60)
        rows = []
        for name, m in self._reg_results.items():
            rows.append({
                "Model": name,
                "Train RMSE": m["train_rmse"],
                "Test RMSE": m["test_rmse"],
                "Train MAE": m["train_mae"],
                "Test MAE": m["test_mae"],
                "Train R2": m["train_r2"],
                "Test R2": m["test_r2"],
            })
        print(pd.DataFrame(rows).to_string(index=False))

        print("\n" + "=" * 60)
        print("SUPERVISED MODELS - CLASSIFICATION RESULTS")
        print("=" * 60)
        rows = []
        for name, m in self._cls_results.items():
            rows.append({
                "Model": name,
                "Train Acc": m["train_accuracy"],
                "Test Acc": m["test_accuracy"],
                "Train F1": m["train_f1"],
                "Test F1": m["test_f1"],
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        return self

    # ------------------------------------------------------------------
    def run_pipeline(self, output_dir: str | None = None) -> "SupervisedModels":
        self.train_all()
        self.plot_results(output_dir)
        self.print_summary()
        return self

    def get_regression_metrics(self) -> dict[str, dict]:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in self._reg_results.items()}

    def get_classification_metrics(self) -> dict[str, dict]:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in self._cls_results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Ensemble models
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleModels:
    """
    Ensemble learning:
    - Random Forest (bagging)  - regressor + classifier
    - LightGBM (boosting)      - regressor + classifier

    Random Forest is fitted on a subsample of the training data
    (default 200 000 rows) to keep runtime practical.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like  (continuous ARR_DELAY)
    rf_n_samples : int
        Training rows used for Random Forest.
    verbose : bool
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_reg=None,
        y_test_reg=None,
        rf_n_samples: int | None = None,
        verbose: bool = True,
    ):
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        self.X_train = np.asarray(X_train, dtype=float)
        self.X_test = np.asarray(X_test, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float)
        self.y_test = np.asarray(y_test, dtype=float)
        self.y_train_cls = make_class_labels(self.y_train)
        self.y_test_cls = make_class_labels(self.y_test)
        self.y_train_reg = np.asarray(y_train_reg if y_train_reg is not None else y_train, dtype=float)
        self.y_test_reg = np.asarray(y_test_reg if y_test_reg is not None else y_test, dtype=float)
        self.rf_n_samples = rf_n_samples
        self.verbose = verbose

        self._reg_results: dict[str, dict] = {}
        self._cls_results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def _subsample_rf(self):
        if self.rf_n_samples is None or self.rf_n_samples >= len(self.X_train):
            # use full training set
            return self.X_train, self.y_train_reg, self.y_train_cls
        rng = np.random.default_rng(48)
        idx = rng.choice(len(self.X_train), self.rf_n_samples, replace=False)
        return self.X_train[idx], self.y_train_reg[idx], self.y_train_cls[idx]

    def _store_reg(self, name: str, model, X_tr, y_tr) -> None:
        tr_pred = model.predict(X_tr)
        te_pred = model.predict(self.X_test)
        self._reg_results[name] = {
            **_reg_metrics(y_tr, tr_pred, "train_"),
            **_reg_metrics(self.y_test_reg, te_pred, "test_"),
            "_model": model,
            "_te_pred": te_pred,
        }

    def _store_cls(self, name: str, model, X_tr, y_tr_cls) -> None:
        tr_pred = model.predict(X_tr)
        te_pred = model.predict(self.X_test)
        self._cls_results[name] = {
            **_cls_metrics(y_tr_cls, tr_pred, "train_"),
            **_cls_metrics(self.y_test_cls, te_pred, "test_"),
            "_model": model,
            "_te_pred": te_pred,
        }

    # ------------------------------------------------------------------
    def train_random_forest(self) -> "EnsembleModels":
        X_sub, y_sub, y_sub_cls = self._subsample_rf()
        n = len(X_sub)

        if self.verbose:
            print(f"[Ensemble] Random Forest Regressor (n={n:,}) ...")
        rfr = RandomForestRegressor(n_estimators=100, max_depth=12,
                                    n_jobs=-1, random_state=48)
        rfr.fit(X_sub, y_sub)
        self._store_reg("Random Forest", rfr, X_sub, y_sub)

        if self.verbose:
            print(f"[Ensemble] Random Forest Classifier (n={n:,}) ...")
        rfc = RandomForestClassifier(n_estimators=100, max_depth=12,
                                     class_weight="balanced",
                                     n_jobs=-1, random_state=48)
        rfc.fit(X_sub, y_sub_cls)
        self._store_cls("Random Forest", rfc, X_sub, y_sub_cls)
        return self

    def train_lightgbm(self) -> "EnsembleModels":
        params = dict(n_estimators=200, max_depth=8, learning_rate=0.05,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      random_state=48, n_jobs=-1, verbose=-1)

        if self.verbose:
            print("[Ensemble] LightGBM Regressor ...")
        lgbr = LGBMRegressor(**params)
        lgbr.fit(self.X_train, self.y_train_reg)
        self._store_reg("LightGBM", lgbr, self.X_train, self.y_train_reg)

        if self.verbose:
            print("[Ensemble] LightGBM Classifier ...")
        lgbc = LGBMClassifier(**params, class_weight="balanced")
        lgbc.fit(self.X_train, self.y_train_cls)
        self._store_cls("LightGBM", lgbc, self.X_train, self.y_train_cls)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _model_dir(base: Path | None, model_name: str) -> Path | None:
        if base is None:
            return None
        d = base / model_name.lower().replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plot_feature_importance(
        self, output_dir: str | None = None, top_n: int = 15
    ) -> "EnsembleModels":
        out = _ensure_dir(output_dir)
        seen: set[str] = set()
        for name, m in {**self._reg_results, **self._cls_results}.items():
            model = m["_model"]
            if not hasattr(model, "feature_importances_"):
                continue
            task = "reg" if name in self._reg_results else "cls"
            key = f"{name}_{task}"
            if key in seen:
                continue
            seen.add(key)

            imp = model.feature_importances_
            model_out = self._model_dir(out, name)
            # ascending so most important is at the top of barh
            indices = np.argsort(imp)[-top_n:]
            vals = imp[indices]
            feat_labels = (
                [self.feature_names[i] for i in indices]
                if self.feature_names else [f"feat_{i}" for i in indices]
            )
            colors = plt.cm.YlOrRd(np.linspace(0.25, 0.9, len(indices)))

            fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))
            bars = ax.barh(range(len(indices)), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(feat_labels, fontsize=9)
            ax.set_xlabel("Importance", fontsize=11)
            ax.set_title(
                f"{name} ({task.upper()}) - Feature Importance (top {top_n})",
                fontsize=12, fontweight="bold",
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8,
                )
            ax.set_xlim(0, max(vals) * 1.15)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            if model_out:
                plt.savefig(model_out / f"{task}_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.show()
        return self

    def plot_results(self, output_dir: str | None = None) -> "EnsembleModels":
        out = _ensure_dir(output_dir)
        for name, m in self._reg_results.items():
            model_out = self._model_dir(out, name)
            path = (model_out / "regression_residuals.png") if model_out else None
            _plot_residuals(self.y_test_reg, m["_te_pred"], f"{name} Regressor - Regression Results", path)
        for name, m in self._cls_results.items():
            model_out = self._model_dir(out, name)
            path = (model_out / "classification_confusion.png") if model_out else None
            _plot_confusion(self.y_test_cls, m["_te_pred"],
                            f"{name} Classifier - Confusion Matrix", path)
        return self

    def print_summary(self) -> "EnsembleModels":
        print("\n" + "=" * 60)
        print("ENSEMBLE MODELS - REGRESSION RESULTS")
        print("=" * 60)
        rows = []
        for name, m in self._reg_results.items():
            rows.append({
                "Model": name,
                "Train RMSE": m["train_rmse"],
                "Test RMSE": m["test_rmse"],
                "Train MAE": m["train_mae"],
                "Test MAE": m["test_mae"],
                "Test R2": m["test_r2"],
            })
        print(pd.DataFrame(rows).to_string(index=False))

        print("\n" + "=" * 60)
        print("ENSEMBLE MODELS - CLASSIFICATION RESULTS")
        print("=" * 60)
        rows = []
        for name, m in self._cls_results.items():
            rows.append({
                "Model": name,
                "Train Acc": m["train_accuracy"],
                "Test Acc": m["test_accuracy"],
                "Train F1": m["train_f1"],
                "Test F1": m["test_f1"],
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        return self

    def run_pipeline(self, output_dir: str | None = None) -> "EnsembleModels":
        self.train_random_forest()
        self.train_lightgbm()
        self.plot_results(output_dir)
        self.plot_feature_importance(output_dir)
        self.print_summary()
        return self

    def get_regression_metrics(self) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in self._reg_results.items()}

    def get_classification_metrics(self) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in self._cls_results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Deep Learning (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """Shared MLP backbone - swap the head for regression vs classification."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepLearningModel:
    """
    Multi-layer Perceptron trained with PyTorch.

    Can perform regression (predict continuous ARR_DELAY) or
    classification (3-class delay category).

    Training is performed on a random subsample of the full training set
    for speed; test evaluation uses a fixed subsample as well.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like  (continuous ARR_DELAY)
    task : {'regression', 'classification'}
    n_train_samples : int
        Subsample size for training (default 200 000).
    n_test_samples : int
        Subsample size for evaluation (default 50 000).
    batch_size : int
    n_epochs : int
    lr : float
    verbose : bool
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        task: str = "regression",
        n_train_samples: int = 200_000,
        n_test_samples: int = 50_000,
        batch_size: int = 1024,
        n_epochs: int = 15,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        self.task = task
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # subsample & prepare arrays
        rng = np.random.default_rng(48)
        n = min(n_train_samples, len(X_train))
        idx = rng.choice(len(X_train), n, replace=False)
        X_tr = np.asarray(X_train, dtype=float)[idx]
        y_tr = np.asarray(y_train, dtype=float)[idx]

        n_te = min(n_test_samples, len(X_test))
        idx_te = rng.choice(len(X_test), n_te, replace=False)
        X_te = np.asarray(X_test, dtype=float)[idx_te]
        y_te = np.asarray(y_test, dtype=float)[idx_te]

        # feature standardisation (fitted on subsample)
        self._scaler = StandardScaler().fit(X_tr)
        X_tr = self._scaler.transform(X_tr)
        X_te = self._scaler.transform(X_te)

        if task == "classification":
            y_tr_cls = make_class_labels(y_tr)
            y_tr_t = torch.from_numpy(y_tr_cls).long()
            y_te_t = torch.from_numpy(make_class_labels(y_te)).long()
            self.y_test_true = make_class_labels(y_te)
            # balanced class weights for CrossEntropyLoss
            cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr_cls)
            self._class_weights = torch.tensor(cw, dtype=torch.float32)
        else:
            y_tr_t = torch.from_numpy(y_tr.reshape(-1, 1)).float()
            y_te_t = torch.from_numpy(y_te.reshape(-1, 1)).float()
            self.y_test_true = y_te
            self._class_weights = None

        self._train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(), y_tr_t
        )
        self._test_X = torch.from_numpy(X_te).float()
        self._test_y = y_te_t

        input_dim = X_tr.shape[1]
        output_dim = 1 if task == "regression" else 3
        self.model = _MLP(input_dim, output_dim).to(self.device)

        self.history: list[float] = []
        self.metrics: dict = {}
        self.y_test_pred: np.ndarray | None = None

    # ------------------------------------------------------------------
    def train(self) -> "DeepLearningModel":
        loader = DataLoader(self._train_ds, batch_size=self.batch_size, shuffle=True)
        if self.task == "regression":
            criterion = nn.MSELoss()
        else:
            w = self._class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=w)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"[DL] Device: {self.device} | Parameters: {n_params:,} | "
                f"Task: {self.task} | Epochs: {self.n_epochs}"
            )

        self.model.train()
        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimiser.zero_grad()
                out = self.model(X_batch)
                if self.task == "classification":
                    loss = criterion(out, y_batch)
                else:
                    loss = criterion(out, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(X_batch)

            avg_loss = epoch_loss / len(self._train_ds)
            self.history.append(avg_loss)
            if self.verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d}/{self.n_epochs}  loss={avg_loss:.4f}")

        return self

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._test_X.to(self.device)).cpu()
            if self.task == "classification":
                self.y_test_pred = out.argmax(dim=1).numpy()
            else:
                self.y_test_pred = out.squeeze().numpy()
        return self.y_test_pred

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        if self.y_test_pred is None:
            self.predict()
        if self.task == "classification":
            self.metrics = _cls_metrics(self.y_test_true, self.y_test_pred)
        else:
            self.metrics = _reg_metrics(self.y_test_true, self.y_test_pred)

        if self.verbose:
            print(f"\n{'=' * 50}\nDeep Learning ({self.task}) RESULTS\n{'=' * 50}")
            for k, v in self.metrics.items():
                print(f"  {k:<22}: {v:.4f}")
            print()
        return self.metrics

    # ------------------------------------------------------------------
    def plot_training_loss(self, output_dir: str | None = None) -> "DeepLearningModel":
        out = _ensure_dir(output_dir)
        epochs = range(1, len(self.history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, self.history, color="#2166ac", lw=2)
        ax.fill_between(epochs, self.history, alpha=0.15, color="#2166ac")
        ax.scatter(epochs, self.history, c=self.history, cmap="YlOrRd_r",
                   s=40, zorder=5, edgecolors="white", lw=0.5)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Training Loss", fontsize=11)
        ax.set_title(
            f"Deep Learning ({self.task}) - Training Loss",
            fontsize=12, fontweight="bold",
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if out:
            path = out / f"dl_{self.task}_training_loss.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"[DL] Training loss plot saved to {path}")
        plt.show()
        return self

    def plot_results(self, output_dir: str | None = None) -> "DeepLearningModel":
        out = _ensure_dir(output_dir)
        if self.task == "classification":
            path = (out / "dl_classification_confusion.png") if out else None
            _plot_confusion(self.y_test_true, self.y_test_pred,
                            "Deep Learning - Confusion Matrix", path)
        else:
            path = (out / "dl_regression_residuals.png") if out else None
            _plot_residuals(self.y_test_true, self.y_test_pred,
                            "Deep Learning - Regression Residuals", path)
        return self

    # ------------------------------------------------------------------
    def run_pipeline(self, output_dir: str | None = None) -> dict:
        self.train()
        self.predict()
        metrics = self.evaluate()
        self.plot_training_loss(output_dir)
        self.plot_results(output_dir)
        return {f"test_{k}": v for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Clustering
# ──────────────────────────────────────────────────────────────────────────────

# Continuous feature names used for clustering (present after DataSplit)
_CLUSTER_FEATURES = [
    "CRS_ELAPSED_TIME", "DISTANCE", "AVG_SPEED",
    "CRS_DEP_TIME_sin", "CRS_DEP_TIME_cos",
    "CRS_ARR_TIME_sin", "CRS_ARR_TIME_cos",
    "FL_MONTH", "FL_DAY_OF_WEEK",
]


class ClusteringModels:
    """
    Unsupervised clustering:
    - KMeans with varying k  (k = 2, 3, 4, 5, 6)
    - DBSCAN with varying eps

    A PCA reduction to 2 components is used for visualisation.
    Silhouette score and Davies-Bouldin index are computed for labelled runs.

    Parameters
    ----------
    X_train : pd.DataFrame or array-like
        Full training feature matrix (columns identified by name if possible).
    n_kmeans_samples : int
        Rows used for KMeans (default 50 000).
    n_dbscan_samples : int
        Rows used for DBSCAN (default 5 000; DBSCAN is O(n2)).
    verbose : bool
    """

    def __init__(
        self,
        X_train,
        n_kmeans_samples: int = 50_000,
        n_dbscan_samples: int = 5_000,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.n_kmeans_samples = n_kmeans_samples
        self.n_dbscan_samples = n_dbscan_samples

        # select clustering features if column names are available
        if isinstance(X_train, pd.DataFrame):
            avail = [c for c in _CLUSTER_FEATURES if c in X_train.columns]
            X_raw = X_train[avail].values.astype(float) if avail else X_train.values.astype(float)
        else:
            X_raw = np.asarray(X_train, dtype=float)

        # standardise before clustering
        scaler = StandardScaler()
        self._X_full = scaler.fit_transform(X_raw)

        # PCA to 2D for visualisation
        self._pca = PCA(n_components=2, random_state=48)
        self._X2d_full = self._pca.fit_transform(self._X_full)

        self._kmeans_results: list[dict] = []
        self._dbscan_results: list[dict] = []

    # ------------------------------------------------------------------
    def _subsample(self, n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._X_full), min(n, len(self._X_full)), replace=False)
        return idx

    # ------------------------------------------------------------------
    def run_kmeans(
        self, k_values: list[int] | None = None
    ) -> "ClusteringModels":
        if k_values is None:
            k_values = [2, 3, 4, 5, 6]

        idx = self._subsample(self.n_kmeans_samples)
        X_sub = self._X_full[idx]
        X2d_sub = self._X2d_full[idx]

        if self.verbose:
            print(f"\n[Clustering] KMeans on {len(X_sub):,} samples ...")

        for k in k_values:
            km = KMeans(n_clusters=k, init="k-means++", n_init=5,
                        max_iter=300, random_state=48)
            labels = km.fit_predict(X_sub)
            n_noise = 0  # KMeans has no noise
            sil = silhouette_score(X_sub, labels, sample_size=5000, random_state=42) \
                if k > 1 else float("nan")
            db = davies_bouldin_score(X_sub, labels)
            inertia = km.inertia_

            self._kmeans_results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "davies_bouldin": db,
                "_labels": labels,
                "_X2d": X2d_sub,
                "_km_model": km,
            })
            if self.verbose:
                print(f"  k={k}: inertia={inertia:.0f}  sil={sil:.4f}  DB={db:.4f}")

        return self

    # ------------------------------------------------------------------
    def run_dbscan(
        self,
        eps_values: list[float] | None = None,
        min_samples: int = 10,
    ) -> "ClusteringModels":
        if eps_values is None:
            eps_values = [0.3, 0.5, 1.0]

        idx = self._subsample(self.n_dbscan_samples, seed=7)
        X_sub = self._X_full[idx]
        X2d_sub = self._X2d_full[idx]

        if self.verbose:
            print(f"\n[Clustering] DBSCAN on {len(X_sub):,} samples ...")

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X_sub)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            # silhouette needs >= 2 clusters and some non-noise points
            unique_non_noise = set(labels) - {-1}
            if len(unique_non_noise) >= 2:
                mask = labels != -1
                sil = silhouette_score(X_sub[mask], labels[mask], sample_size=min(2000, mask.sum()), random_state=42)
            else:
                sil = float("nan")

            self._dbscan_results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": 100 * n_noise / len(labels),
                "silhouette": sil,
                "_labels": labels,
                "_X2d": X2d_sub,
                "_db_model": db,
            })
            if self.verbose:
                print(
                    f"  eps={eps}: clusters={n_clusters}  noise={n_noise} "
                    f"({100*n_noise/len(labels):.1f}%)  sil={sil:.4f}"
                )

        return self

    # ------------------------------------------------------------------
    def plot_elbow(self, output_dir: str | None = None) -> "ClusteringModels":
        if not self._kmeans_results:
            return self
        out = _ensure_dir(output_dir)
        k_vals = [r["k"] for r in self._kmeans_results]
        inertias = [r["inertia"] for r in self._kmeans_results]
        sils = [r["silhouette"] for r in self._kmeans_results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(k_vals, inertias, color="#2166ac", lw=2)
        axes[0].fill_between(k_vals, inertias, alpha=0.12, color="#2166ac")
        for k, v in zip(k_vals, inertias):
            axes[0].plot(k, v, "o", ms=9, color="#2166ac", zorder=5)
            axes[0].text(k, v, f"  {v:,.0f}", va="bottom", fontsize=8)
        axes[0].set_xlabel("k", fontsize=11)
        axes[0].set_ylabel("Inertia (WCSS)", fontsize=11)
        axes[0].set_title("KMeans - Elbow Curve", fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)

        sil_min, sil_max = min(sils), max(sils)
        sil_span = sil_max - sil_min if sil_max > sil_min else 1.0
        sil_colors = plt.cm.RdYlGn([(s - sil_min) / sil_span for s in sils])
        axes[1].plot(k_vals, sils, color="#888", lw=1.5, zorder=1)
        for k, s, c in zip(k_vals, sils, sil_colors):
            axes[1].plot(k, s, "o", ms=12, color=c, zorder=5)
            axes[1].text(k, s, f"  {s:.3f}", va="bottom", fontsize=8)
        axes[1].set_xlabel("k", fontsize=11)
        axes[1].set_ylabel("Silhouette Score", fontsize=11)
        axes[1].set_title("KMeans - Silhouette Score", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if out:
            path = out / "kmeans_elbow.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"[Clustering] Elbow plot saved to {path}")
        plt.show()
        return self

    def plot_clusters(self, output_dir: str | None = None) -> "ClusteringModels":
        out = _ensure_dir(output_dir)

        # KMeans - plot best silhouette k
        if self._kmeans_results:
            best = max(self._kmeans_results, key=lambda r: r["silhouette"])
            self._scatter_clusters(
                best["_X2d"], best["_labels"],
                title=f"KMeans k={best['k']} - PCA projection",
                noise_label=None,
                save_path=(out / f"kmeans_k{best['k']}_clusters.png") if out else None,
            )

        # DBSCAN - plot each eps
        for r in self._dbscan_results:
            self._scatter_clusters(
                r["_X2d"], r["_labels"],
                title=f"DBSCAN eps={r['eps']} - {r['n_clusters']} clusters",
                noise_label=-1,
                save_path=(out / f"dbscan_eps{r['eps']}_clusters.png") if out else None,
            )
        return self

    @staticmethod
    def _scatter_clusters(X2d, labels, title, noise_label=None, save_path=None):
        unique = sorted(set(labels))
        palette = sns.color_palette("tab10", n_colors=max(len(unique), 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            color = "lightgrey" if lbl == noise_label else palette[i % len(palette)]
            name = "Noise" if lbl == noise_label else f"Cluster {lbl}"
            ax.scatter(X2d[mask, 0], X2d[mask, 1], s=6, alpha=0.5,
                       color=color, label=name)
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(markerscale=2, fontsize=9, bbox_to_anchor=(1, 1))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    def print_summary(self) -> "ClusteringModels":
        if self._kmeans_results:
            print("\n" + "=" * 60)
            print("CLUSTERING - KMeans RESULTS")
            print("=" * 60)
            rows = [{"k": r["k"], "Inertia": r["inertia"],
                     "Silhouette": r["silhouette"],
                     "Davies-Bouldin": r["davies_bouldin"]}
                    for r in self._kmeans_results]
            print(pd.DataFrame(rows).to_string(index=False))

        if self._dbscan_results:
            print("\n" + "=" * 60)
            print("CLUSTERING - DBSCAN RESULTS")
            print("=" * 60)
            rows = [{"eps": r["eps"], "Clusters": r["n_clusters"],
                     "Noise %": f"{r['noise_pct']:.1f}",
                     "Silhouette": r["silhouette"]}
                    for r in self._dbscan_results]
            print(pd.DataFrame(rows).to_string(index=False))
        print()
        return self

    def run_pipeline(
        self,
        k_values: list[int] | None = None,
        eps_values: list[float] | None = None,
        output_dir: str | None = None,
    ) -> "ClusteringModels":
        self.run_kmeans(k_values)
        self.run_dbscan(eps_values)
        self.plot_elbow(output_dir)
        self.plot_clusters(output_dir)
        self.print_summary()
        return self

    def get_kmeans_metrics(self) -> list[dict]:
        return [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in self._kmeans_results]

    def get_dbscan_metrics(self) -> list[dict]:
        return [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in self._dbscan_results]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Model comparison
# ──────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    """
    Aggregate metrics from all models and produce a unified comparison.

    Usage
    -----
    cmp = ModelComparison()
    cmp.add_regression("Linear Regression", metrics_dict)
    cmp.add_classification("Logistic Regression", metrics_dict)
    cmp.print_all()
    cmp.plot_comparison(output_dir="Output files/models")
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._reg: list[dict] = []
        self._cls: list[dict] = []
        self._clu: list[dict] = []

    # ------------------------------------------------------------------
    def add_regression(self, name: str, metrics: dict) -> "ModelComparison":
        self._reg.append({"Model": name, **metrics})
        return self

    def add_classification(self, name: str, metrics: dict) -> "ModelComparison":
        self._cls.append({"Model": name, **metrics})
        return self

    def add_clustering(self, name: str, metrics: dict) -> "ModelComparison":
        self._clu.append({"Model": name, **metrics})
        return self

    # ------------------------------------------------------------------
    def get_regression_table(self) -> pd.DataFrame:
        cols = ["Model", "test_rmse", "test_mae", "test_r2"]
        df = pd.DataFrame(self._reg)
        return df[[c for c in cols if c in df.columns]].rename(columns={
            "test_rmse": "RMSE", "test_mae": "MAE", "test_r2": "R2"
        }).sort_values("RMSE")

    def get_classification_table(self) -> pd.DataFrame:
        cols = ["Model", "test_accuracy", "test_f1", "test_precision", "test_recall"]
        df = pd.DataFrame(self._cls)
        return df[[c for c in cols if c in df.columns]].rename(columns={
            "test_accuracy": "Accuracy", "test_f1": "F1",
            "test_precision": "Precision", "test_recall": "Recall",
        }).sort_values("Accuracy", ascending=False)

    def get_clustering_table(self) -> pd.DataFrame:
        return pd.DataFrame(self._clu)

    # ------------------------------------------------------------------
    def print_all(self) -> "ModelComparison":
        if self._reg:
            print("\n" + "=" * 65)
            print("MODEL COMPARISON - REGRESSION (test set)")
            print("=" * 65)
            print(self.get_regression_table().to_string(index=False))

        if self._cls:
            print("\n" + "=" * 65)
            print("MODEL COMPARISON - CLASSIFICATION (test set)")
            print("=" * 65)
            print(self.get_classification_table().to_string(index=False))

        if self._clu:
            print("\n" + "=" * 65)
            print("MODEL COMPARISON - CLUSTERING")
            print("=" * 65)
            print(self.get_clustering_table().to_string(index=False))

        print()
        return self

    # ------------------------------------------------------------------
    def plot_comparison(self, output_dir: str | None = None) -> "ModelComparison":
        out = _ensure_dir(output_dir)

        def _labeled_barh(ax, models, values, cmap, label,
                          reverse_cmap=False, xlim_min=None):
            norm_vals = np.array(values, dtype=float)
            span = norm_vals.max() - norm_vals.min()
            t = (norm_vals - norm_vals.min()) / (span if span > 0 else 1.0)
            if reverse_cmap:
                t = 1.0 - t
            colors = plt.cm.get_cmap(cmap)(0.3 + t * 0.6)
            bars = ax.barh(models, norm_vals, color=colors, edgecolor="white", height=0.6)

            x_lo = xlim_min if xlim_min is not None else 0.0
            x_hi = norm_vals.max()
            x_range = max(x_hi - x_lo, 1e-9)

            for bar, v in zip(bars, norm_vals):
                if v >= 0:
                    ax.text(v + x_range * 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{v:.4f}", va="center", ha="left", fontsize=9)
                else:
                    ax.text(v - x_range * 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{v:.4f}", va="center", ha="right", fontsize=9)

            if x_lo < 0:
                ax.axvline(0, color="black", lw=0.8, alpha=0.4, zorder=0)

            ax.set_xlim(x_lo - x_range * 0.02, x_hi + x_range * 0.22)
            ax.set_xlabel(label, fontsize=11)
            ax.grid(axis="x", alpha=0.3)

        if self._reg and len(self._reg) > 1:
            df = self.get_regression_table()
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.6 + 1)))
            rmse_min = float(df["RMSE"].min())
            r2_min   = float(df["R2"].min())
            # Zoom RMSE axis so small differences between models are visible
            _labeled_barh(axes[0], df["Model"], df["RMSE"], "RdYlGn",
                          "RMSE (lower is better, axis zoomed)",
                          reverse_cmap=True, xlim_min=rmse_min * 0.97)
            axes[0].set_title("Regression - Test RMSE", fontsize=12, fontweight="bold")
            # Allow negative R2 values by starting axis below zero
            _labeled_barh(axes[1], df["Model"], df["R2"], "RdYlGn",
                          "R2 (higher is better)",
                          xlim_min=min(0.0, r2_min - abs(r2_min) * 0.15))
            axes[1].set_title("Regression - Test R2", fontsize=12, fontweight="bold")
            plt.tight_layout()
            if out:
                path = out / "comparison_regression.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                if self.verbose:
                    print(f"[Comparison] Regression chart saved to {path}")
            plt.show()

        if self._cls and len(self._cls) > 1:
            df = self.get_classification_table()
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.6 + 1)))
            _labeled_barh(axes[0], df["Model"], df["Accuracy"],
                          "Blues", "Accuracy (higher is better)")
            axes[0].set_title("Classification - Test Accuracy", fontsize=12, fontweight="bold")
            _labeled_barh(axes[1], df["Model"], df["F1"],
                          "Oranges", "F1 Weighted (higher is better)")
            axes[1].set_title("Classification - Test F1", fontsize=12, fontweight="bold")
            plt.tight_layout()
            if out:
                path = out / "comparison_classification.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                if self.verbose:
                    print(f"[Comparison] Classification chart saved to {path}")
            plt.show()

        return self


# ──────────────────────────────────────────────────────────────────────────────
# 7. ModelTraining - backward compatibility (LightGBM regression)
# ──────────────────────────────────────────────────────────────────────────────

class ModelTraining:
    """
    LightGBM regression wrapper kept for backward compatibility.

    For the full Part-2 model suite use the classes above directly
    (KNNScratch, SupervisedModels, EnsembleModels, DeepLearningModel,
    ClusteringModels, ModelComparison).
    """

    def __init__(self, X_train, X_test, y_train, y_test, verbose: bool = True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.verbose = verbose
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.metrics = {}

    def train(self):
        if self.verbose:
            print("Training LightGBM regression model ...")
        self.model = LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=48, n_jobs=-1, verbose=-1,
        )
        self.model.fit(self.X_train, self.y_train)
        if self.verbose:
            print("Model training completed.\n")
        return self

    def predict(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        return self

    def evaluate(self):
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        self.metrics = {
            "train_mse": train_mse, "train_rmse": np.sqrt(train_mse),
            "train_mae": mean_absolute_error(self.y_train, self.y_train_pred),
            "train_r2": r2_score(self.y_train, self.y_train_pred),
            "test_mse": test_mse, "test_rmse": np.sqrt(test_mse),
            "test_mae": mean_absolute_error(self.y_test, self.y_test_pred),
            "test_r2": r2_score(self.y_test, self.y_test_pred),
        }
        if self.verbose:
            print("=" * 60 + "\nMODEL EVALUATION METRICS\n" + "=" * 60)
            for k, v in self.metrics.items():
                print(f"  {k:<18}: {v:.4f}")
            print("=" * 60 + "\n")
        return self.metrics

    def plot_predictions_vs_actual(self, output_dir=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.5, s=20)
        mn = min(self.y_test.min(), self.y_test_pred.min())
        mx = max(self.y_test.max(), self.y_test_pred.max())
        plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
        plt.xlabel("Actual Arrival Delay (min)")
        plt.ylabel("Predicted Arrival Delay (min)")
        plt.title("LightGBM: Predicted vs Actual")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            plt.savefig(p / "predictions_vs_actual_lightgbm.png", dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"Plot saved to {p / 'predictions_vs_actual_lightgbm.png'}")
        plt.show()
        return self

    def plot_residuals(self, output_dir=None):
        residuals = np.asarray(self.y_test) - self.y_test_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(self.y_test_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(0, color="r", ls="--", lw=2)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted"); axes[0].grid(alpha=0.3)
        axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals"); axes[1].set_title("Distribution of Residuals")
        plt.tight_layout()
        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            plt.savefig(p / "residuals_lightgbm.png", dpi=300, bbox_inches="tight")
        plt.show()
        return self

    def plot_feature_importance(self, top_n=20, output_dir=None):
        imp = self.model.feature_importances_
        names = list(self.X_train.columns) if hasattr(self.X_train, "columns") else [f"f{i}" for i in range(len(imp))]
        indices = np.argsort(imp)[-top_n:][::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), imp[indices], color="steelblue")
        plt.yticks(range(len(indices)), [names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"LightGBM: Top {top_n} Feature Importance")
        plt.tight_layout()
        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            plt.savefig(p / "feature_importance_lightgbm.png", dpi=300, bbox_inches="tight")
        plt.show()
        return self

    def run_pipeline(self, output_dir=None):
        self.train()
        self.predict()
        metrics = self.evaluate()
        if output_dir:
            self.plot_predictions_vs_actual(output_dir)
            self.plot_residuals(output_dir)
            self.plot_feature_importance(output_dir=output_dir)
        return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 8. Model saving utility
# ──────────────────────────────────────────────────────────────────────────────

def _write_model_card(
    root: Path,
    feature_names: list | None,
    sup: "SupervisedModels | None",
    ens: "EnsembleModels | None",
    dl_c: "DeepLearningModel | None",
    dl_r: "DeepLearningModel | None",
    knn_c: "KNNScratch | None",
    knn_r: "KNNScratch | None",
    clust: "ClusteringModels | None",
) -> None:
    """Write MODEL_CARD.md to root directory."""
    lines: list[str] = []
    a = lines.append

    a("# Model Card - Flight Delay Prediction\n\n")
    a("## Overview\n\n")
    a("Models trained to predict US domestic flight arrival delays using the\n")
    a("Flight Delay and Cancellation Dataset (2019-2023), ~2.9 million flights.\n")
    a("Two tasks are supported: regression (delay in minutes) and classification\n")
    a("(3-class delay category).\n\n")

    a("## Dataset\n\n")
    a("- **Source**: Flight Delay and Cancellation Dataset (2019-2023)\n")
    a("- **Rows**: ~2.9 million flights after cleaning\n")
    a("- **Split**: 80% train / 20% test (stratified by year)\n\n")

    a("## Target Variable\n\n")
    a("- **Regression**: `ARR_DELAY` (minutes), winsorized at 99th percentile\n")
    a("  (~189 min) to reduce outlier influence.\n")
    a("- **Classification** (3 classes):\n")
    a("  - `0` On-time: ARR_DELAY < 15 min\n")
    a("  - `1` Short delay: 15 <= ARR_DELAY <= 30 min\n")
    a("  - `2` Long delay: ARR_DELAY > 30 min\n\n")

    if feature_names:
        a("## Input Features\n\n")
        a(f"The models expect **{len(feature_names)} features** in this order:\n\n")
        for i, fn in enumerate(feature_names):
            a(f"  {i + 1:2d}. `{fn}`\n")
        a("\n`DISTANCE`, `CRS_ELAPSED_TIME`, and `AVG_SPEED` are StandardScaler-normalised.\n")
        a("All categorical features use integer (ordinal) encoding.\n\n")

    a("## Models\n\n")

    # KNN
    a("### 1. k-Nearest Neighbours (from scratch)\n\n")
    a("- **Files**: `knn/knn_classification.joblib`, `knn/knn_regression.joblib`\n")
    a("- **Implementation**: Pure NumPy; vectorised L2 distance\n")
    a("- **Hyperparameters**: k=5, balanced class weights, z-score normalisation\n")
    a("- **Training subset**: 20,000 samples (O(n*m) scaling constraint)\n")
    a("- **Note**: Metrics are indicative only (subsampled 5,000-row test set)\n")
    if knn_c is not None and knn_c.metrics:
        a(f"- **Classification test**: Accuracy={knn_c.metrics.get('accuracy', float('nan')):.4f}"
          f", F1={knn_c.metrics.get('f1', float('nan')):.4f}\n")
    if knn_r is not None and knn_r.metrics:
        a(f"- **Regression test**: RMSE={knn_r.metrics.get('rmse', float('nan')):.4f}"
          f", R2={knn_r.metrics.get('r2', float('nan')):.4f}\n")
    a("\n```python\n")
    a("import json, numpy as np\n")
    a("from model_training import KNNScratch\n")
    a("params = json.loads(open('knn/knn_classification_params.json').read())\n")
    a("data   = np.load('knn/knn_classification_arrays.npz')\n")
    a("knn = KNNScratch(k=params['k'], task=params['task'])\n")
    a("knn._X_tr = data['X_tr']; knn._y_tr = data['y_tr']\n")
    a("knn._mean = data['mean'];  knn._std  = data['std']\n")
    a("knn._class_weights = data['class_weights'] if 'class_weights' in data else None\n")
    a("knn.metrics = params['metrics']\n")
    a("preds = knn.predict(X_new)  # numpy array (n_samples, n_features)\n")
    a("```\n\n")

    # Decision Tree
    a("### 2. Decision Tree (scikit-learn)\n\n")
    a("- **Files**: `supervised/decision_tree_cls.joblib`,\n")
    a("  `supervised/decision_tree_reg.joblib`\n")
    a("- **Hyperparameters**: max_depth=12, class_weight='balanced' (cls), random_state=48\n")
    a("- **Training subset**: Full dataset (~2.33M samples)\n")
    if sup is not None:
        cls_m = sup._cls_results.get("Decision Tree (cls)", {})
        reg_m = sup._reg_results.get("Decision Tree (reg)", {})
        if cls_m:
            a(f"- **Classification test**: Accuracy={cls_m.get('test_accuracy', float('nan')):.4f}"
              f", F1={cls_m.get('test_f1', float('nan')):.4f}\n")
        if reg_m:
            a(f"- **Regression test**: RMSE={reg_m.get('test_rmse', float('nan')):.4f}"
              f", R2={reg_m.get('test_r2', float('nan')):.4f}\n")
    a("\n```python\n")
    a("import joblib\n")
    a("dt_cls = joblib.load('supervised/decision_tree_cls.joblib')\n")
    a("preds = dt_cls.predict(X_new)  # returns 0, 1, or 2\n")
    a("```\n\n")

    # Random Forest
    a("### 3. Random Forest (scikit-learn)\n\n")
    a("- **Files**: `ensemble/random_forest_classifier.joblib`,\n")
    a("  `ensemble/random_forest_regressor.joblib`\n")
    a("- **Hyperparameters**: n_estimators=100, max_depth=12,\n")
    a("  class_weight='balanced', n_jobs=-1, random_state=48\n")
    a("- **Training subset**: Full dataset (~2.33M samples)\n")
    if ens is not None:
        cls_m = ens._cls_results.get("Random Forest", {})
        reg_m = ens._reg_results.get("Random Forest", {})
        if cls_m:
            a(f"- **Classification test**: Accuracy={cls_m.get('test_accuracy', float('nan')):.4f}"
              f", F1={cls_m.get('test_f1', float('nan')):.4f}\n")
        if reg_m:
            a(f"- **Regression test**: RMSE={reg_m.get('test_rmse', float('nan')):.4f}"
              f", R2={reg_m.get('test_r2', float('nan')):.4f}\n")
    a("\n```python\n")
    a("import joblib\n")
    a("rf_cls = joblib.load('ensemble/random_forest_classifier.joblib')\n")
    a("preds = rf_cls.predict(X_new)  # returns 0, 1, or 2\n")
    a("```\n\n")

    # LightGBM
    a("### 4. LightGBM\n\n")
    a("- **Files**: `ensemble/lightgbm_classifier.joblib`,\n")
    a("  `ensemble/lightgbm_regressor.joblib`\n")
    a("- **Hyperparameters**: n_estimators=200, max_depth=8, learning_rate=0.05,\n")
    a("  num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=48\n")
    a("- **Training subset**: Full dataset (~2.33M samples)\n")
    if ens is not None:
        cls_m = ens._cls_results.get("LightGBM", {})
        reg_m = ens._reg_results.get("LightGBM", {})
        if cls_m:
            a(f"- **Classification test**: Accuracy={cls_m.get('test_accuracy', float('nan')):.4f}"
              f", F1={cls_m.get('test_f1', float('nan')):.4f}\n")
        if reg_m:
            a(f"- **Regression test**: RMSE={reg_m.get('test_rmse', float('nan')):.4f}"
              f", R2={reg_m.get('test_r2', float('nan')):.4f}\n")
    a("\n```python\n")
    a("import joblib\n")
    a("lgbm_cls = joblib.load('ensemble/lightgbm_classifier.joblib')\n")
    a("preds = lgbm_cls.predict(X_new)  # returns 0, 1, or 2\n")
    a("```\n\n")

    # Deep Learning
    a("### 5. Deep Learning MLP (PyTorch)\n\n")
    a("- **Classifier files**: `deep_learning/mlp_classifier_weights.pt`,\n")
    a("  `deep_learning/mlp_classifier_config.json`,\n")
    a("  `deep_learning/mlp_classifier_scaler.joblib`\n")
    a("- **Regressor files**: same with `_regressor_` prefix\n")
    a("- **Architecture**: input -> 256 -> 128 -> 64 -> output, ReLU + Dropout(0.3)\n")
    a("- **Training**: 15 epochs, batch_size=1024, lr=1e-3, Adam optimizer\n")
    a("- **Training subset**: 500,000 samples (CPU constraint)\n")
    if dl_c is not None and dl_c.metrics:
        a(f"- **Classification test**: Accuracy={dl_c.metrics.get('accuracy', float('nan')):.4f}"
          f", F1={dl_c.metrics.get('f1', float('nan')):.4f}\n")
    if dl_r is not None and dl_r.metrics:
        a(f"- **Regression test**: RMSE={dl_r.metrics.get('rmse', float('nan')):.4f}"
          f", R2={dl_r.metrics.get('r2', float('nan')):.4f}\n")
    a("\n```python\n")
    a("import json, joblib, torch\n")
    a("from model_training import _MLP\n")
    a("cfg = json.loads(open('deep_learning/mlp_classifier_config.json').read())\n")
    a("model = _MLP(cfg['input_dim'], cfg['output_dim'])\n")
    a("model.load_state_dict(torch.load('deep_learning/mlp_classifier_weights.pt',\n")
    a("                                 map_location='cpu'))\n")
    a("model.eval()\n")
    a("scaler = joblib.load('deep_learning/mlp_classifier_scaler.joblib')\n")
    a("X_scaled = scaler.transform(X_new)\n")
    a("with torch.no_grad():\n")
    a("    logits = model(torch.tensor(X_scaled, dtype=torch.float32))\n")
    a("    preds = logits.argmax(1).numpy()\n")
    a("```\n\n")

    # KMeans
    a("### 6. KMeans Clustering (scikit-learn)\n\n")
    a("- **Files**: `clustering/kmeans_k2.joblib` ... `clustering/kmeans_k6.joblib`\n")
    a("- **Hyperparameters**: init='k-means++', n_init=5, max_iter=300, random_state=48\n")
    a("- **Training subset**: 200,000 samples\n")
    a("- **Best model by silhouette**: k=3 (score ~0.251)\n")
    a("- **Features used**: CRS_ELAPSED_TIME, DISTANCE, AVG_SPEED,\n")
    a("  CRS_DEP_TIME_sin/cos, CRS_ARR_TIME_sin/cos, FL_MONTH, FL_DAY_OF_WEEK\n")
    a("- **Note**: Features must be StandardScaler-normalised before prediction.\n")
    a("\n```python\n")
    a("import joblib\n")
    a("kmeans = joblib.load('clustering/kmeans_k3.joblib')\n")
    a("labels = kmeans.predict(X_new_scaled)  # 9-feature StandardScaler output\n")
    a("```\n\n")

    # DBSCAN
    a("### 7. DBSCAN Clustering (scikit-learn)\n\n")
    a("- **Files**: `clustering/dbscan_eps*.joblib`\n")
    a("- **Hyperparameters**: min_samples=10, n_jobs=-1\n")
    a("- **Training subset**: 5,000 samples (O(n^2) constraint)\n")
    a("- **Note**: DBSCAN does not support `predict()` on unseen data.\n")
    a("  Saved for reference and analysis only.\n\n")

    a("## Limitations\n\n")
    a("- Models predict delay from scheduled flight attributes only; they have no\n")
    a("  access to real-time weather, actual departure delay, or in-flight data.\n")
    a("- kNN and Deep Learning models use training subsamples and show weaker\n")
    a("  performance than full-data tree models.\n")
    a("- All categorical encodings must match those generated by\n")
    a("  `DataSplit.export_encoding_mappings()` (see `encoding_mappings.csv`).\n\n")

    a("## Authors\n\n")
    a("Alexis Barros (2045719) & Vitor Remesso (2050519)\n")
    a("Data Science Project 2025/2026\n")

    (root / "MODEL_CARD.md").write_text("".join(lines))
    print(f"[ModelSaver] MODEL_CARD.md written to {root / 'MODEL_CARD.md'}")


def save_models(
    save_dir: "str | Path",
    knn_cls: "KNNScratch | None" = None,
    knn_reg: "KNNScratch | None" = None,
    supervised: "SupervisedModels | None" = None,
    ensemble: "EnsembleModels | None" = None,
    dl_cls: "DeepLearningModel | None" = None,
    dl_reg: "DeepLearningModel | None" = None,
    clustering: "ClusteringModels | None" = None,
    feature_names: "list[str] | None" = None,
) -> Path:
    """
    Serialize all trained model objects to `save_dir`, write MODEL_CARD.md,
    and return the path to a zip archive of the directory.

    Parameters
    ----------
    save_dir : str or Path
        Destination directory (created if needed).
    knn_cls, knn_reg : KNNScratch, optional
    supervised : SupervisedModels, optional
    ensemble : EnsembleModels, optional
    dl_cls, dl_reg : DeepLearningModel, optional
    clustering : ClusteringModels, optional
    feature_names : list[str], optional
        Ordered feature names matching the training columns.

    Returns
    -------
    Path  path to the created zip archive
    """
    import joblib
    import json
    import zipfile

    root = _ensure_dir(save_dir)
    saved: list[str] = []

    # --- KNN -----------------------------------------------------------------
    # KNNScratch is a pure-Python/numpy class. joblib.dump would pickle the
    # class type, which fails if model_training was reloaded after the instance
    # was created (class identity mismatch). Save arrays + params separately.
    knn_dir = root / "knn"
    for tag, knn_obj in [("knn_classification", knn_cls), ("knn_regression", knn_reg)]:
        if knn_obj is None:
            continue
        knn_dir.mkdir(exist_ok=True)
        params = {
            "k": knn_obj.k,
            "task": knn_obj.task,
            "n_train_samples": knn_obj.n_train_samples,
            "n_test_samples": knn_obj.n_test_samples,
            "random_state": knn_obj.random_state,
            "class_weight": knn_obj.class_weight,
            "verbose": knn_obj.verbose,
            "metrics": knn_obj.metrics,
        }
        (knn_dir / f"{tag}_params.json").write_text(json.dumps(params, indent=2))
        arrays: dict = {}
        for attr in ("_X_tr", "_y_tr", "_mean", "_std"):
            val = getattr(knn_obj, attr, None)
            if val is not None:
                arrays[attr.lstrip("_")] = val
        if knn_obj._class_weights is not None:
            arrays["class_weights"] = knn_obj._class_weights
        np.savez(str(knn_dir / f"{tag}_arrays.npz"), **arrays)
        saved += [
            str(knn_dir / f"{tag}_params.json"),
            str(knn_dir / f"{tag}_arrays.npz"),
        ]

    # --- Supervised ----------------------------------------------------------
    if supervised is not None:
        sup_dir = root / "supervised"
        sup_dir.mkdir(exist_ok=True)
        for name, m in supervised._cls_results.items():
            fname = (name.lower()
                     .replace(" ", "_")
                     .replace("(", "")
                     .replace(")", "") + ".joblib")
            p = sup_dir / fname
            joblib.dump(m["_model"], p)
            saved.append(str(p))
        for name, m in supervised._reg_results.items():
            fname = (name.lower()
                     .replace(" ", "_")
                     .replace("(", "")
                     .replace(")", "") + ".joblib")
            p = sup_dir / fname
            joblib.dump(m["_model"], p)
            saved.append(str(p))

    # --- Ensemble ------------------------------------------------------------
    if ensemble is not None:
        ens_dir = root / "ensemble"
        ens_dir.mkdir(exist_ok=True)
        for name, m in ensemble._cls_results.items():
            fname = name.lower().replace(" ", "_") + "_classifier.joblib"
            p = ens_dir / fname
            joblib.dump(m["_model"], p)
            saved.append(str(p))
        for name, m in ensemble._reg_results.items():
            fname = name.lower().replace(" ", "_") + "_regressor.joblib"
            p = ens_dir / fname
            joblib.dump(m["_model"], p)
            saved.append(str(p))

    # --- Deep Learning -------------------------------------------------------
    dl_dir = root / "deep_learning"
    if dl_cls is not None:
        dl_dir.mkdir(exist_ok=True)
        cfg = {
            "input_dim": dl_cls.model.net[0].in_features,
            "output_dim": 3,
            "task": "classification",
            "architecture": "MLP 256->128->64->output, ReLU + Dropout(0.3)",
            "n_epochs": dl_cls.n_epochs,
            "lr": dl_cls.lr,
            "batch_size": dl_cls.batch_size,
        }
        (dl_dir / "mlp_classifier_config.json").write_text(json.dumps(cfg, indent=2))
        torch.save(dl_cls.model.state_dict(), dl_dir / "mlp_classifier_weights.pt")
        joblib.dump(dl_cls._scaler, dl_dir / "mlp_classifier_scaler.joblib")
        saved += [
            str(dl_dir / "mlp_classifier_config.json"),
            str(dl_dir / "mlp_classifier_weights.pt"),
            str(dl_dir / "mlp_classifier_scaler.joblib"),
        ]
    if dl_reg is not None:
        dl_dir.mkdir(exist_ok=True)
        cfg = {
            "input_dim": dl_reg.model.net[0].in_features,
            "output_dim": 1,
            "task": "regression",
            "architecture": "MLP 256->128->64->output, ReLU + Dropout(0.3)",
            "n_epochs": dl_reg.n_epochs,
            "lr": dl_reg.lr,
            "batch_size": dl_reg.batch_size,
        }
        (dl_dir / "mlp_regressor_config.json").write_text(json.dumps(cfg, indent=2))
        torch.save(dl_reg.model.state_dict(), dl_dir / "mlp_regressor_weights.pt")
        joblib.dump(dl_reg._scaler, dl_dir / "mlp_regressor_scaler.joblib")
        saved += [
            str(dl_dir / "mlp_regressor_config.json"),
            str(dl_dir / "mlp_regressor_weights.pt"),
            str(dl_dir / "mlp_regressor_scaler.joblib"),
        ]

    # --- Clustering ----------------------------------------------------------
    if clustering is not None:
        clu_dir = root / "clustering"
        clu_dir.mkdir(exist_ok=True)
        for r in clustering._kmeans_results:
            if "_km_model" in r:
                p = clu_dir / f"kmeans_k{r['k']}.joblib"
                joblib.dump(r["_km_model"], p)
                saved.append(str(p))
        for r in clustering._dbscan_results:
            if "_db_model" in r:
                p = clu_dir / f"dbscan_eps{r['eps']}.joblib"
                joblib.dump(r["_db_model"], p)
                saved.append(str(p))

    # --- Feature names -------------------------------------------------------
    if feature_names is not None:
        p = root / "feature_names.json"
        p.write_text(json.dumps(feature_names, indent=2))
        saved.append(str(p))

    # --- Model card ----------------------------------------------------------
    _write_model_card(
        root, feature_names,
        supervised, ensemble,
        dl_cls, dl_reg,
        knn_cls, knn_reg,
        clustering,
    )

    # --- Zip -----------------------------------------------------------------
    zip_path = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(root.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(root.parent))

    print(f"[ModelSaver] {len(saved)} files saved to {root}")
    print(f"[ModelSaver] Zip archive: {zip_path}")
    return zip_path