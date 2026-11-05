"""Cross-validation, metric (macro-F1) en Bayesiaanse vergelijking met ROPE (baycomp).

Per fold wordt alles wat van data leert (BPE-tokenizer, TF-IDF, model) alleen op het
trainingsdeel van die fold gefit, zodat de validatie-fold echt ongezien blijft.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from part1.analysis import BenavoliCorrelatedT
from part1.experiment import train_tokenizer

TOKENIZER_VOCAB_SIZE = 1000
TOKENIZER_MIN_FREQUENCY = 2
ROPE = 0.01  # verschil in macro-F1 dat we als "praktisch gelijk" beschouwen, zie motivatie in het notebook
DECISION_THRESHOLD = 0.95  # vanaf deze kans trekken we een conclusie, anders "onbeslist"


def _run_fold(fold: dict, encoder_factories: dict, model_runners: dict, num_classes: int) -> dict:
    """Train tokenizer, encoders en alle modellen op één fold; geef macro-F1 per systeem."""
    training_set, validation_set = fold["training"], fold["validation"]
    train_texts = [text for text, _ in training_set]
    gold = [label for _, label in validation_set]

    tokenizer = train_tokenizer(train_texts, TOKENIZER_VOCAB_SIZE, TOKENIZER_MIN_FREQUENCY)
    scores = {}
    for enc_name, make_encoder in encoder_factories.items():
        encoder = make_encoder(tokenizer, train_texts)
        for model_name, run_model in model_runners.items():
            preds = run_model(training_set, validation_set, encoder, num_classes)
            scores[f"{model_name}_{enc_name}"] = f1_score(gold, preds, average="macro")
    return scores


def run_cross_validation(cv_runs: list, encoder_factories: dict, model_runners: dict, num_classes: int) -> dict:
    """Macro-F1 per systeem over alle runs en folds: {systeem: [f1 run0-fold0, run0-fold1, ...]}.

    cv_runs is de uitvoer van src.data.make_cv_splits (een lijst met per seed een dict van folds).
    """
    fold_scores = [
        _run_fold(run[fold_id], encoder_factories, model_runners, num_classes)
        for run in cv_runs
        for fold_id in sorted(run)
    ]
    return {name: [scores[name] for scores in fold_scores] for name in fold_scores[0]}


def summary_table(f1_scores: dict, encoders: list, models: list) -> pd.DataFrame:
    """Gemiddelde macro-F1 als tabel: rijen = modellen, kolommen = representaties."""
    return pd.DataFrame(
        {enc: [np.mean(f1_scores[f"{model}_{enc}"]) for model in models] for enc in encoders},
        index=models,
    )


def majority_baseline_f1(cv_runs: list) -> float:
    """Gemiddelde macro-F1 van een 'model' dat altijd de meest voorkomende klasse van de trainingsfold voorspelt."""
    scores = []
    for run in cv_runs:
        for fold in run.values():
            majority = Counter(label for _, label in fold["training"]).most_common(1)[0][0]
            gold = [label for _, label in fold["validation"]]
            scores.append(f1_score(gold, [majority] * len(gold), average="macro"))
    return float(np.mean(scores))


def _verdict(p_a: float, p_tie: float, p_b: float, sys_a: str, sys_b: str) -> str:
    if p_a >= DECISION_THRESHOLD:
        return f"{sys_a} beter"
    if p_b >= DECISION_THRESHOLD:
        return f"{sys_b} beter"
    if p_tie >= DECISION_THRESHOLD:
        return "praktisch gelijk"
    return "onbeslist"


def _compare_pair(f1_scores: dict, sys_a: str, sys_b: str, runs: int, rope: float) -> dict:
    comparison = BenavoliCorrelatedT(f1_scores[sys_a], f1_scores[sys_b], runs=runs, names=[sys_a, sys_b])
    p_a, p_tie, p_b = comparison.probs(rope)
    return {
        "A": sys_a, "B": sys_b,
        "gem. verschil (B-A)": comparison.mean_diff,
        "P(A beter)": p_a, "P(gelijk)": p_tie, "P(B beter)": p_b,
        "conclusie": _verdict(p_a, p_tie, p_b, sys_a, sys_b),
    }


def compare_systems(f1_scores: dict, systems: list, runs: int, rope: float = ROPE) -> pd.DataFrame:
    """Bayesiaanse correlated t-test voor elk paar systemen: P(A beter), P(praktisch gelijk), P(B beter)."""
    rows = [_compare_pair(f1_scores, a, b, runs, rope) for a, b in combinations(systems, 2)]
    return pd.DataFrame(rows).round(3)


def plot_comparison(f1_scores: dict, sys_a: str, sys_b: str, runs: int, rope: float = ROPE):
    """Posterior van het verschil (B-A) met de ROPE, zoals in de tutorial."""
    return BenavoliCorrelatedT(f1_scores[sys_a], f1_scores[sys_b], runs=runs, names=[sys_a, sys_b]).plot(rope)
