"""Cross-validation voor de neurale modellen van tutorial 2 (embed-and-pool, LSTM) op onze stance-data.

Gebruikt part2.experiment.cross_validation, met dezelfde folds/seeds als project 1 (src.data.make_cv_splits),
zodat de scores vergelijkbaar zijn. Geeft macro-F1 per fold terug, net als src.evaluate.
"""

import contextlib
import io
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from part2 import experiment
from part2.clslib import PretrainedEmbeddings, Tokenizer
from src.data import LABEL_NAMES


@contextlib.contextmanager
def silenced():
    """Alle uitvoer weg, ook wat buiten Python om geschreven wordt (bv. de voortgang van HF-tokenizers)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def cross_validate(cv_runs: list, tokenizer: Tokenizer, emb_model: PretrainedEmbeddings, enc_type: str,
                   enc_kwargs: dict, regressor_kwargs: dict, opt_kwargs: dict, save_to: Path,
                   verbose: bool = False) -> list[float]:
    """Train en evalueer op alle runs x folds; geeft de macro-F1 per fold (bv. 3 seeds x 5 folds = 15 scores)."""
    scores = []
    for run_id, run in enumerate(cv_runs):
        torch.manual_seed(run_id)  # part2 zet zelf geen seed: zo geeft elke Run All dezelfde scores
        # part2.experiment.train past de kwargs-dicts aan (padding_idx, num_classes), dus geef kopieën mee
        with contextlib.nullcontext() if verbose else silenced():
            result = experiment.cross_validation(
                outcome_space=LABEL_NAMES, data_splits=run, tokenizer=tokenizer, emb_model=emb_model,
                enc_type=enc_type, enc_kwargs=deepcopy(enc_kwargs), regressor_kwargs=deepcopy(regressor_kwargs),
                opt_kwargs=deepcopy(opt_kwargs), save_to=str(Path(save_to) / f"seed-{run_id}"),
            )
        for fold_id in sorted(run):
            gold = [label for _, label in run[fold_id]["validation"]]
            scores.append(f1_score(gold, result["predictions"][fold_id], average="macro"))
    return scores
