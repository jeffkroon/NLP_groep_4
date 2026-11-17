"""Hulpfuncties om een embeddingmodel tegen onze data te houden (project 2, stap 1 en 2)."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def load_vocab(path: Path) -> set[str]:
    """Alleen de woorden uit een GloVe .txt-bestand (veel sneller dan alle vectoren laden)."""
    with open(path, encoding="utf-8") as f:
        return {line.split(" ", 1)[0] for line in f}


def is_word(token: str) -> bool:
    """Echt woord (letters, eventueel met apostrof), dus geen leesteken of speciaal token als <user>."""
    return token.replace("'", "").isalpha()


def coverage(texts: list[str], tokenizer: Callable[[str], list[str]], vocab: set[str], top_missing: int = 8) -> dict:
    """Welk deel van de tokens in onze teksten staat in de vocabulaire van het model?

    - tokens gedekt: alle voorkomens geteld (een vaak gebruikt woord telt zwaar)
    - unieke tokens gedekt: elk verschillend token telt één keer (laat zeldzame woorden zien)
    """
    counts = Counter(token for text in texts for token in tokenizer(text))
    known = {token for token in counts if token in vocab}
    return {
        "tokens gedekt": round(sum(counts[t] for t in known) / sum(counts.values()), 3),
        "unieke tokens gedekt": round(len(known) / len(counts), 3),
        "vaakst ontbrekend": [t for t, _ in counts.most_common() if t not in known and is_word(t)][:top_missing],
    }


def coverage_table(texts: list[str], setups: dict) -> pd.DataFrame:
    """setups: {naam: (tokenizer, vocab)} -> één rij per opzet."""
    return pd.DataFrame({name: coverage(texts, tok, vocab) for name, (tok, vocab) in setups.items()}).T


def write_subset(source: Path, target: Path, words: set[str]) -> int:
    """Schrijf alleen de regels van `words` uit een GloVe .txt naar `target`; geeft het aantal woorden terug.

    Zo is de embeddingtabel ~10k woorden i.p.v. 1.2M: sneller laden en modelbestanden van KB's i.p.v. 240 MB.
    Geen datalek: het zijn voorgetrainde vectoren, we kiezen alleen welke woorden we opzoeken.
    """
    kept = 0
    with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        for line in src:
            if line.split(" ", 1)[0] in words:
                dst.write(line)
                kept += 1
    return kept
