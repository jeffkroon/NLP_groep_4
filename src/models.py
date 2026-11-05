"""Modellen: log-linear, MLP met 1 verborgen laag en MLP met 2 verborgen lagen (onze derde architectuur).

Beide MLP's worden met exact dezelfde procedure getraind (train_mlp): zelfde optimizer, loss,
dropout, epochs en keuze van het beste epoch. Het enige verschil is het aantal verborgen lagen,
zodat een verschil in score ook echt aan die extra laag toe te schrijven is.
"""


import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn
from torch.utils.data import DataLoader

from part1 import loglin
from part1.clslib import Encoder
from part1.smlp import (
    SparseDataset,
    SparseMLP,
    evaluate,
    sparse_collate,
    train_one_epoch,
)

# Zelfde standaardwaarden als part1.smlp.train_neural_network_model
MLP_DEFAULTS = {
    "hidden_dim": 128,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 20,
}
SECOND_HIDDEN_DIM = 64
LOGISTIC_MAX_ITER = 1000
TORCH_SEED = 0


class SparseMLP2(nn.Module):
    """Zoals part1.smlp.SparseMLP, maar met een tweede verborgen laag (ReLU + dropout)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1,
                 second_hidden_dim: int = SECOND_HIDDEN_DIM):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(second_hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1),
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(x, self.weight) + self.bias
        x = self.dropout(self.activation(x))
        return self.output(self.hidden2(x))


def _make_loader(pairs: list, encoder: Encoder, batch_size: int, shuffle: bool) -> DataLoader:
    X = csr_matrix([encoder(text) for text, _ in pairs])
    y = [int(label) for _, label in pairs]
    return DataLoader(SparseDataset(X, y), batch_size=batch_size, shuffle=shuffle, collate_fn=sparse_collate)


def train_mlp(model_class, training_set: list, validation_set: list, encoder: Encoder,
              num_classes: int, **overrides) -> np.ndarray:
    """Train een MLP zoals part1.smlp en geef de validatievoorspellingen van het beste epoch (laagste val loss)."""
    args = {**MLP_DEFAULTS, **overrides}
    torch.manual_seed(TORCH_SEED)  # zelfde startgewichten en batchvolgorde voor elke run

    train_loader = _make_loader(training_set, encoder, args["batch_size"], shuffle=True)
    val_loader = _make_loader(validation_set, encoder, args["batch_size"], shuffle=False)

    model = model_class(input_dim=len(encoder), hidden_dim=args["hidden_dim"],
                        num_classes=num_classes, dropout=args["dropout"])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"],
                                  weight_decay=args["weight_decay"])

    best_loss, best_preds = float("inf"), None
    for _ in range(args["num_epochs"]):
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics, preds = evaluate(model, val_loader, criterion)
        if val_metrics["loss"] <= best_loss:
            best_loss, best_preds = val_metrics["loss"], preds
    return best_preds


def run_logistic(training_set: list, validation_set: list, encoder: Encoder, num_classes: int) -> np.ndarray:
    model = loglin.fit(training_set, encoder, max_iter=LOGISTIC_MAX_ITER)
    return np.asarray(loglin.predict([text for text, _ in validation_set], model, encoder))


def run_mlp(training_set: list, validation_set: list, encoder: Encoder, num_classes: int) -> np.ndarray:
    return train_mlp(SparseMLP, training_set, validation_set, encoder, num_classes)


def run_mlp_2hidden(training_set: list, validation_set: list, encoder: Encoder, num_classes: int) -> np.ndarray:
    return train_mlp(SparseMLP2, training_set, validation_set, encoder, num_classes)


# Naam -> functie(training_set, validation_set, encoder, num_classes) -> voorspellingen op validation_set
MODEL_RUNNERS = {
    "logistic": run_logistic,
    "mlp": run_mlp,
    "mlp_2hidden": run_mlp_2hidden,
}
