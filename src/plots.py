"""Overzichtsfiguren voor de vergelijking van systemen (matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Kleuren (gevalideerd palet: categorisch in vaste volgorde, sequentieel blauw, divergerend blauw <-> rood)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
DIVERGING_A, DIVERGING_TIE, DIVERGING_B = "#2a78d6", "#d9d8d4", "#e34948"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MIN_LABEL_SHARE = 0.08  # percentages kleiner dan dit krijgen geen label in de balk


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)


def plot_score_heatmap(f1_scores: dict, encoders: list, models: list, baseline: float):
    """Heatmap: modellen x representaties, gemiddelde macro-F1 ± standaardafwijking over de folds."""
    means = np.array([[np.mean(f1_scores[f"{m}_{e}"]) for e in encoders] for m in models])
    stds = np.array([[np.std(f1_scores[f"{m}_{e}"]) for e in encoders] for m in models])

    fig, ax = plt.subplots(figsize=(1.8 * len(encoders) + 2, 0.7 * len(models) + 1.5))
    image = ax.imshow(means, cmap=SEQUENTIAL_BLUE, aspect="auto")
    best = np.unravel_index(means.argmax(), means.shape)
    for i in range(len(models)):
        for j in range(len(encoders)):
            dark_cell = means[i, j] > means.min() + 0.6 * (means.max() - means.min())
            weight = "bold" if (i, j) == best else "normal"
            ax.text(j, i, f"{means[i, j]:.3f}\n± {stds[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if dark_cell else TEXT_PRIMARY, fontweight=weight)

    ax.set_xticks(range(len(encoders)), encoders)
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04)
    colorbar.set_label("macro-F1", color=TEXT_SECONDARY)
    colorbar.outline.set_visible(False)
    ax.set_title(f"Mean macro-F1 per system (majority baseline: {baseline:.3f})",
                 loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    return fig, ax


def plot_bayes_comparisons(comparisons: pd.DataFrame, title: str, strip: str = ""):
    """Gestapelde balk per paar: P(A beter) | P(praktisch gelijk) | P(B beter), met de conclusie ernaast.

    strip: tekst die uit de systeemnamen wordt gehaald (bv. "_bow" als alle systemen bow gebruiken).
    """
    rows = comparisons.iloc[::-1].reset_index(drop=True)
    short = lambda name: name.replace(strip, "") if strip else name
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.4))
    segments = [("P(A better)", DIVERGING_A, "white"), ("P(equal)", DIVERGING_TIE, TEXT_PRIMARY),
                ("P(B better)", DIVERGING_B, "white")]
    left = np.zeros(len(rows))
    for column, color, text_color in segments:
        widths = rows[column].to_numpy()
        ax.barh(range(len(rows)), widths, left=left, color=color, edgecolor="white", linewidth=2, height=0.7)
        for y, (x0, w) in enumerate(zip(left, widths)):
            if w >= MIN_LABEL_SHARE:
                ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=9, color=text_color)
        left = left + widths

    ax.set_yticks(range(len(rows)), [f"{short(a)}  vs  {short(b)}" for a, b in zip(rows["A"], rows["B"])])
    for y, verdict in enumerate(rows["verdict"]):
        ax.text(1.02, y, short(verdict), va="center", fontsize=9, color=TEXT_PRIMARY, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in segments]
    ax.legend(handles, ["A better", "practically equal (ROPE)", "B better"], ncol=3, frameon=False,
              loc="upper left", bbox_to_anchor=(0, -0.08 if len(rows) > 2 else -0.2), fontsize=9)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    return fig, ax
