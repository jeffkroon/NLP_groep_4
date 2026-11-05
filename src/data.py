"""Dataset laden en splitsen, zoals in part1/tutorial.ipynb (load_dataset + cross_validation_split).

Dataset volgens de course-site (Stance Detection): TweetEval, https://huggingface.co/datasets/cardiffnlp/tweet_eval
"""

from datasets import DatasetDict, concatenate_datasets, load_dataset

from part1.experiment import cross_validation_split

DATASET_ID = "cardiffnlp/tweet_eval"
STANCE_TOPICS = ("abortion", "atheism", "climate", "feminist", "hillary")
LABEL_NAMES = ("none", "against", "favor")

SEEDS = (0, 3, 5)  # elke seed = een andere verdeling in folds; meer runs = betrouwbaardere vergelijking
N_FOLDS = 5

# Bij alle topics samen zetten we het topic voor de tweet, want "against" betekent per topic iets anders
TOPIC_TAG = "[{topic}]"

# Artefact uit SemEval-2016 taak 6: staat in elke tweet en draagt geen informatie
ARTIFACT_TAG = "#SemST"


def _remove_artifact(example: dict) -> dict:
    return {"text": example["text"].replace(ARTIFACT_TAG, "").strip()}


def load_stance(topic: str) -> DatasetDict:
    """Laad train/validation/test voor één stance-topic, zonder het #SemST-artefact."""
    if topic not in STANCE_TOPICS:
        raise ValueError(f"Onbekend topic '{topic}', kies uit {STANCE_TOPICS}")
    return load_dataset(DATASET_ID, f"stance_{topic}").map(_remove_artifact)


def _tag_with_topic(dataset: DatasetDict, topic: str) -> DatasetDict:
    tag = TOPIC_TAG.format(topic=topic)
    return dataset.map(lambda example: {"text": f"{tag} {example['text']}", "topic": topic})


def load_stance_all(topics: tuple = STANCE_TOPICS) -> DatasetDict:
    """Laad alle stance-topics samen (per split aan elkaar geplakt), met het topic als tag voor elke tweet."""
    per_topic = [_tag_with_topic(load_stance(topic), topic) for topic in topics]
    return DatasetDict({
        split: concatenate_datasets([dataset[split] for dataset in per_topic])
        for split in per_topic[0]
    })


def make_cv_splits(dataset: DatasetDict, seeds=SEEDS, k=N_FOLDS) -> list:
    """Cross-validation folds op de train-split; de test-split blijft ongezien.

    Index de resultaten als splits[run][fold]["training" | "validation"], net als in de tutorial.
    """
    pairs = list(zip(dataset["train"]["text"], dataset["train"]["label"]))
    return [cross_validation_split(pairs=pairs, k=k, seed=seed) for seed in seeds]
