import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    Audio,
    load_from_disk,
)
from config import TrainingConfig
from utils import detect_language_from_path, get_num_cpus


def load_and_standardize_dataset(
    config_or_path,
    is_local: bool = False,
    split: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
):
    if is_local:
        print(f"Loading local dataset from {config_or_path}...")
        try:
            ds = load_from_disk(config_or_path)
            if isinstance(ds, DatasetDict):
                if "train" in ds:
                    ds = ds["train"]
                elif len(ds.keys()) > 0:
                    first_key = list(ds.keys())[0]
                    ds = ds[first_key]
        except Exception as e:
            print(f"Error load_from_disk {config_or_path}: {e}")
            return None
    else:
        print(f"Loading online dataset {config_or_path['path']}...")
        ds = load_dataset(
            config_or_path["path"],
            split=split or config_or_path.get("split"),
            name=config_or_path.get("name"),
            subset=config_or_path.get("subset"),
            trust_remote_code=trust_remote_code or config_or_path.get("trust_remote_code", False),
            token=token or config_or_path.get("token", None),
        )

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    possible_text_cols = [
        "sentence", "text", "text_original", "transcription",
        "transcribe", "label", "content", "segment_text", "transcript",
    ]
    target_col = None
    for col in possible_text_cols:
        if col in ds.column_names:
            target_col = col
            break

    if target_col and target_col != "sentence":
        ds = ds.rename_column(target_col, "sentence")
    elif not target_col:
        raise ValueError(f"Could not find a text column. Available: {ds.column_names}")

    detected_lang = detect_language_from_path(config_or_path)

    if "language" in ds.column_names:
        ds = ds.remove_columns(["language"])
    ds = ds.add_column("language", [detected_lang] * len(ds))
    print(f"   -> Detected language: {detected_lang}")

    keep_cols = ["audio", "sentence", "language"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    return ds


def load_all_datasets(config: TrainingConfig):
    loaded_datasets = []

    for cfg in config.online_train_datasets:
        ds = load_and_standardize_dataset(cfg)
        if ds is not None:
            loaded_datasets.append(ds)

    for p in config.local_train_datasets:
        if os.path.exists(p):
            ds = load_and_standardize_dataset(p, is_local=True)
            if ds is not None:
                loaded_datasets.append(ds)

    if not loaded_datasets:
        raise ValueError("No datasets loaded! Check dataset paths.")

    train_dataset = concatenate_datasets(loaded_datasets).shuffle(seed=42)

    print("Preprocessing text labels for train dataset...")
    train_dataset = train_dataset.map(
        prepare_dataset, batched=True, batch_size=1000, num_proc=4, desc="Train",
        disable_progress_bar=True,
    )
    return train_dataset


def load_all_test_datasets(config: TrainingConfig):
    test_datasets = []

    for cfg in config.online_test_datasets:
        try:
            ds = load_and_standardize_dataset(cfg)
            if ds is not None:
                test_datasets.append(ds)
                print(f"Loaded online test: {cfg['path']}")
        except Exception as e:
            print(f"Error loading {cfg['path']}: {e}")

    for p in config.local_test_datasets:
        if os.path.exists(p):
            ds = load_and_standardize_dataset(p, is_local=True)
            if ds is not None:
                test_datasets.append(ds)
                print(f"Loaded local test: {p}")

    if not test_datasets:
        raise ValueError("No test datasets loaded!")

    test_ds = concatenate_datasets(test_datasets)

    if config.max_test_samples and len(test_ds) > config.max_test_samples:
        test_ds = test_ds.select(range(config.max_test_samples))

    test_ds = test_ds.map(
        prepare_dataset, batched=True, batch_size=1000, num_proc=4, desc="Test",
        disable_progress_bar=True,
    )
    return test_ds


def prepare_dataset(batch):
    from utils import clean_text
    output = {}
    output["sentence_cleaned"] = [clean_text(s) for s in batch["sentence"]]
    return output


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    max_label_length: int = 448

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"]["array"] for f in features]
        input_features = self.processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt"
        ).input_features

        label_texts = [f["sentence_cleaned"] for f in features]
        languages = [f.get("language", "vi") for f in features]

        labels_batch = []
        for lang, text in zip(languages, label_texts):
            self.processor.tokenizer.set_prefix_tokens(language=lang, task="transcribe")
            encoded = self.processor.tokenizer(
                text, truncation=True, max_length=self.max_label_length + 1
            ).input_ids
            labels_batch.append(encoded)

        max_label_length = max(len(l) for l in labels_batch)
        padded_labels = [l + [-100] * (max_label_length - len(l)) for l in labels_batch]
        labels = torch.tensor(padded_labels)

        bos_token_id = self.processor.tokenizer.bos_token_id
        if (labels[:, 0] == bos_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }

