"""Tweets tokeniseren zoals Stanford deed voor GloVe Twitter.

Nagebouwd naar Stanfords preprocess-twitter.rb: URL's, @users, getallen, smileys en hashtags worden
vervangen door speciale tokens die in de GloVe Twitter-vocabulaire staan (<url>, <user>, <hashtag>, ...).
Zonder deze stap wordt bv. "#feminism" een onbekend woord, terwijl "<hashtag> feminism" wel bekend is.
"""

import re

EYES = r"[8:=;]"
NOSE = r"['`\-]?"

# Volgorde is belangrijk: eerst URL's en users, dan smileys, dan hashtags/getallen/hoofdletters.
_SUBSTITUTIONS = [
    (re.compile(r"https?://\S+|www\.\S+"), " <url> "),
    (re.compile(r"@\w+"), " <user> "),
    (re.compile(rf"{EYES}{NOSE}[)dD]+|[(dD]+{NOSE}{EYES}"), " <smile> "),
    (re.compile(rf"{EYES}{NOSE}[pP]+"), " <lolface> "),
    (re.compile(rf"{EYES}{NOSE}\(+|\)+{NOSE}{EYES}"), " <sadface> "),
    (re.compile(rf"{EYES}{NOSE}[/|l*]"), " <neutralface> "),
    (re.compile(r"<3"), " <heart> "),
    (re.compile(r"/"), " / "),
    (re.compile(r"[-+]?[.\d]*\d+[:,.\d]*"), " <number> "),
]
_HASHTAG = re.compile(r"#(\w+)")
_REPEATED_PUNCT = re.compile(r"([!?.]){2,}")
_ELONGATED = re.compile(r"\b(\S*?)(\w)\2{2,}\b")
_ALLCAPS = re.compile(r"\b([A-Z]{2,})\b")
# Samentrekkingen splitsen zoals in de GloVe-vocabulaire: don't -> do n't, it's -> it 's, can't -> ca n't
_NEGATION = re.compile(r"(\w+)n't\b")
_CLITIC = re.compile(r"(\w+)('s|'m|'re|'ll|'ve|'d)\b")
_TOKEN = re.compile(r"<\w+>|n't|'\w+|\w+|[^\w\s]")


def _hashtag(match: re.Match) -> str:
    body = match.group(1)
    if body.isupper():
        return f" <hashtag> {body} <allcaps> "
    # CamelCase splitsen: #StopTheHate -> stop the hate
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", body) or [body]
    return " <hashtag> " + " ".join(words) + " "


def preprocess(text: str) -> str:
    """Pas de GloVe Twitter-regels toe en geef de tekst in kleine letters terug."""
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    text = _HASHTAG.sub(_hashtag, text)
    text = _REPEATED_PUNCT.sub(r" \1 <repeat> ", text)
    text = _ELONGATED.sub(r"\1\2 <elong> ", text)
    text = _ALLCAPS.sub(lambda m: f"{m.group(1)} <allcaps>", text)
    text = text.lower()
    text = _NEGATION.sub(r"\1 n't", text)
    return _CLITIC.sub(r"\1 \2", text)


def tokenize(text: str) -> list[str]:
    """Tweet -> lijst tokens die aansluiten bij de GloVe Twitter-vocabulaire."""
    return _TOKEN.findall(preprocess(text))
