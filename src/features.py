"""Documentrepresentaties: token-indicators, bag-of-words en TF-IDF (onze derde representatie).

Elke encoder zet een tekst om in een vector met één dimensie per token in de vocabulaire,
zodat ze uitwisselbaar zijn in part1.loglin en in de MLP's uit src/models.py.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer

from part1.clslib import Encoder


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagOfWords(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of token counts"""
        token_ids = self.tokenizer.encode(x).ids

        # Same as TokenIndicator, but we keep duplicate ids and count them,
        # so a token that appears 3 times gets value 3 instead of 1
        vector = np.bincount(token_ids, minlength=len(self)).astype(float)
        return vector


class TfidfEncoder(Encoder):
    """TF-IDF over BPE-tokens (Mink).

    De IDF-gewichten worden geschat op `training_texts`; geef dus alleen de trainingsdata van de fold mee.
    """

    def __init__(self, tokenizer: Tokenizer, training_texts: list[str]):
        self.tokenizer = tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None,
            lowercase=False,
            dtype=np.float32,
        )
        self.vectorizer.fit(training_texts)

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    def __len__(self) -> int:
        """Dimensionality of the fitted TF-IDF representation."""
        return len(self.vectorizer.vocabulary_)

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a dense vector of BPE-token TF-IDF weights."""
        return self.vectorizer.transform([x]).toarray()[0]


# Naam -> functie die met de tokenizer en trainingsteksten van één fold een encoder bouwt
ENCODER_FACTORIES = {
    "indicator": lambda tokenizer, texts: TokenIndicator(tokenizer),
    "bow": lambda tokenizer, texts: BagOfWords(tokenizer),
    "tfidf": lambda tokenizer, texts: TfidfEncoder(tokenizer, texts),
}
