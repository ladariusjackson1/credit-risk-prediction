"""Charts for the credit study. Figures are written for the README, not shown live."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from credit_risk import config

MODEL_COLORS = {
    "logistic_regression": "#1f4e79",
    "random_forest": "#2a9d8f",
    "gradient_boosting": "#e76f51",
}
INK = "#1f2933"
MUTED = "#52606d"
GRID = "#e4e7eb"


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d9e2ec",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """Mean cross-validated ROC-AUC and average precision, with the fold spread."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    order = summary.copy()
    labels = [config.DISPLAY_NAMES.get(name, name) for name in order["model"]]
    colors = [MODEL_COLORS.get(name, "#334e68") for name in order["model"]]
    y_pos = np.arange(len(order))
    panels = (
        ("roc_auc_mean", "roc_auc_std", "ROC-AUC", "Cross-validated ROC-AUC"),
        (
            "average_precision_mean",
            "average_precision_std",
            "Average precision",
            "Cross-validated average precision",
        ),
    )
    for axis, (mean_col, std_col, xlabel, title) in zip(axes, panels, strict=True):
        means = order[mean_col].to_numpy(dtype=float)
        spread = order[std_col].fillna(0).to_numpy(dtype=float)
        axis.barh(y_pos, means, xerr=spread, color=colors, height=0.62, capsize=3)
        axis.set_yticks(y_pos, labels)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.set_xlim(0.0, 1.0)
        if "ROC" in title:
            axis.axvline(0.5, color="#9aa5b1", linestyle="--", linewidth=1)
        for position, value in zip(y_pos, means, strict=True):
            axis.text(min(value + 0.02, 0.98), position, f"{value:.3f}", va="center", color=INK)
    fig.suptitle("Model comparison on the training folds", fontsize=14, fontweight="bold")
    return _save(fig, path)


def save_roc_curves(y_true, probabilities: dict[str, np.ndarray], path: Path) -> Path:
    _apply_style()
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    actual = np.asarray(y_true).astype(int)
    for name, scores in probabilities.items():
        fpr, tpr, _ = roc_curve(actual, scores)
        score = auc(fpr, tpr)
        axis.plot(
            fpr,
            tpr,
            color=MODEL_COLORS.get(name, INK),
            linewidth=2,
            label=f"{config.DISPLAY_NAMES.get(name, name)}  {score:.3f}",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1, label="No skill")
    axis.set_xlabel("Share of paying applicants the model would deny")
    axis.set_ylabel("Share of defaulters the model would deny")
    axis.set_title("Holdout ROC curves")
    axis.legend(frameon=False, loc="lower right")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    return _save(fig, path)


def save_precision_recall_curves(y_true, probabilities: dict[str, np.ndarray], path: Path) -> Path:
    _apply_style()
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    actual = np.asarray(y_true).astype(int)
    base_rate = float(actual.mean())
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(actual, scores)
        axis.plot(
            recall,
            precision,
            color=MODEL_COLORS.get(name, INK),
            linewidth=2,
            label=config.DISPLAY_NAMES.get(name, name),
        )
    axis.axhline(
        base_rate, linestyle="--", color="#9aa5b1", linewidth=1, label=f"Base rate {base_rate:.2f}"
    )
    axis.set_xlabel("Recall: share of defaulters denied")
    axis.set_ylabel("Precision: defaulters among denials")
    axis.set_title("Holdout precision-recall curves")
    axis.legend(frameon=False, loc="upper right")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    return _save(fig, path)


def save_confusion_matrices(y_true, proba, threshold: float, path: Path) -> Path:
    """Side-by-side counts at 0.50 and at the cost-minimizing cutoff."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    for axis, cutoff, title in zip(
        axes,
        (0.5, float(threshold)),
        ("Score cutoff 0.50", f"Cost-minimizing cutoff {threshold:.2f}"),
        strict=True,
    ):
        decision = (scores >= cutoff).astype(int)
        matrix = np.array(
            [
                [
                    int(((actual == 0) & (decision == 0)).sum()),
                    int(((actual == 0) & (decision == 1)).sum()),
                ],
                [
                    int(((actual == 1) & (decision == 0)).sum()),
                    int(((actual == 1) & (decision == 1)).sum()),
                ],
            ]
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks([0, 1], ["Approve", "Deny"])
        axis.set_yticks([0, 1], ["Would pay", "Would default"])
        axis.set_xlabel("Decision")
        axis.set_ylabel("Outcome")
        axis.set_title(title)
        ceiling = max(int(matrix.max()), 1)
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] > ceiling * 0.55 else INK
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Holdout decisions", fontsize=14, fontweight="bold")
    return _save(fig, path)


def save_threshold_cost(sweep: pd.DataFrame, chosen: float, baselines: dict, path: Path) -> Path:
    """Out-of-fold cost against the cutoff, with the two naive policies."""
    _apply_style()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        sweep["threshold"],
        sweep["expected_cost"],
        color="#1f4e79",
        linewidth=2,
        label="Model policy",
    )
    axis.axhline(
        baselines["approve_all"],
        color="#9aa5b1",
        linestyle="--",
        linewidth=1,
        label="Approve everyone",
    )
    axis.axhline(
        baselines["deny_all"],
        color="#e07a3d",
        linestyle="--",
        linewidth=1,
        label="Deny everyone",
    )
    axis.axvline(chosen, color="#e76f51", linewidth=1.4, label=f"Chosen cutoff {chosen:.2f}")
    axis.set_xlabel("Deny if estimated default probability is at least this cutoff")
    axis.set_ylabel("Expected cost on training out-of-fold scores")
    axis.set_title("Where the desk should set the cutoff")
    axis.legend(frameon=False)
    axis.set_xlim(0, 1)
    return _save(fig, path)


def save_policy_costs(test_table: pd.DataFrame, baselines: dict, selected: str, path: Path) -> Path:
    """Illustrative holdout dollars for naive policies and each tuned model."""
    _apply_style()
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    labels = ["Approve all", "Deny all"]
    values = [baselines["approve_all_usd"], baselines["deny_all_usd"]]
    colors = ["#9aa5b1", "#e07a3d"]
    ordered = test_table.copy()
    ordered["model"] = pd.Categorical(
        ordered["model"], categories=list(config.MODEL_NAMES), ordered=True
    )
    ordered = ordered.sort_values("model")
    for _, row in ordered.iterrows():
        labels.append(config.DISPLAY_NAMES.get(row["model"], row["model"]))
        values.append(float(row["illustrative_cost_usd"]))
        colors.append("#e76f51" if row["model"] == selected else "#1f4e79")
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors, width=0.72)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Illustrative holdout cost ($)")
    axis.set_title("Cost of each policy on the locked holdout")
    ceiling = max(values) if values else 1
    for position, value in zip(positions, values, strict=True):
        axis.text(
            position, value + ceiling * 0.02, f"{value:,.0f}", ha="center", va="bottom", fontsize=9
        )
    axis.set_ylim(0, ceiling * 1.18)
    return _save(fig, path)


def save_importance(table: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    _apply_style()
    subset = table.head(top_n).iloc[::-1]
    fig, axis = plt.subplots(figsize=(7.4, 5.4))
    axis.barh(
        [_pretty(name) for name in subset["feature"]],
        subset["importance_mean"],
        xerr=subset["importance_std"],
        color="#1f4e79",
        height=0.7,
        capsize=2,
    )
    axis.set_xlabel("Drop in holdout ROC-AUC when the column is shuffled")
    axis.set_title("Permutation importance")
    axis.axvline(0, color=MUTED, linewidth=0.8)
    return _save(fig, path)


def save_score_distribution(y_true, proba, threshold: float, path: Path) -> Path:
    _apply_style()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    bins = np.linspace(0, 1, 21)
    axis.hist(
        scores[actual == 0],
        bins=bins,
        alpha=0.75,
        color="#2a9d8f",
        label="Would pay",
        density=True,
    )
    axis.hist(
        scores[actual == 1],
        bins=bins,
        alpha=0.65,
        color="#e76f51",
        label="Would default",
        density=True,
    )
    axis.axvline(threshold, color=INK, linewidth=1.4, label=f"Cutoff {threshold:.2f}")
    axis.set_xlabel("Estimated probability of default")
    axis.set_ylabel("Density")
    axis.set_title("Holdout scores by outcome")
    axis.legend(frameon=False)
    return _save(fig, path)


def save_calibration(y_true, proba, path: Path, n_bins: int = 8) -> Path:
    _apply_style()
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    bins = int(min(n_bins, max(3, len(actual) // 15)))
    fraction, mean_predicted = calibration_curve(actual, scores, n_bins=bins, strategy="quantile")
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1, label="Perfect calibration"
    )
    axis.plot(
        mean_predicted, fraction, marker="o", color="#1f4e79", linewidth=2, label="Selected model"
    )
    axis.set_xlabel("Mean predicted probability in the bin")
    axis.set_ylabel("Observed default rate")
    axis.set_title("Holdout calibration")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def save_segment_rates(
    categorical: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical_feature: str,
    numeric_feature: str,
    base_rate: float,
    path: Path,
) -> Path:
    """Default rate for the strongest categorical column and the strongest numeric column."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    panels = (
        (axes[0], categorical, categorical_feature, "Category"),
        (axes[1], numeric, numeric_feature, "Quantile band"),
    )
    for axis, table, feature, kind in panels:
        subset = table.loc[table["feature"] == feature].sort_values("default_rate")
        if subset.empty:
            axis.set_title(_pretty(feature))
            axis.text(0.5, 0.5, "Not enough variation", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = [str(level) for level in subset["level"]]
        rates = subset["default_rate"].to_numpy(dtype=float)
        axis.barh(labels, rates, color="#1f4e79", height=0.7)
        axis.axvline(
            base_rate,
            color="#e76f51",
            linestyle="--",
            linewidth=1.2,
            label=f"Book rate {base_rate:.0%}",
        )
        axis.set_xlim(0, max(0.75, float(rates.max()) + 0.08))
        axis.set_xlabel("Default rate")
        axis.set_title(f"{_pretty(feature)} ({kind.lower()})")
        for index, (rate, count) in enumerate(zip(rates, subset["n"], strict=True)):
            axis.text(
                rate + 0.01,
                index,
                f"{rate:.0%}  n={int(count)}",
                va="center",
                fontsize=8,
                color=MUTED,
            )
        axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Where defaults actually concentrate", fontsize=14, fontweight="bold")
    return _save(fig, path)
