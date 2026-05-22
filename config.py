from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class TrainingConfig:
    # Model
    model_name_or_path: str = "openai/whisper-medium"
    output_dir: str = "weights/whisper-medium-bilingual-vi-en"

    # Languages
    languages: List[str] = field(default_factory=lambda: ["vi", "en"])
    task: str = "transcribe"

    # Training hyperparameters
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    num_train_epochs: int = 2
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 25
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 3
    max_label_length: int = 448
    generation_max_length: int = 440
    generation_num_beams: int = 1

    # Seq2SeqTrainingArguments extras
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    predict_with_generate: bool = True
    remove_unused_columns: bool = False
    label_names: str = '["labels"]'
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    report_to: str = "wandb"
    prediction_loss_only: bool = False
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: str = '{"use_reentrant": false}'

    # Generation config
    no_repeat_ngram_size: int = 5
    condition_on_previous_text: bool = False
    predict_timestamps: bool = False
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.4

    # Data
    eval_accumulation_steps: int = 1
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = 2000
    dataset_loading_num_proc: int = 4

    # Augmentation
    apply_augmentation: bool = True
    augmentation_noise_path: str = ""
    augmentation_ir_path: str = ""

    # Resume
    resume_from_checkpoint: bool = True

    # WandB
    wandb_project: str = "whisper-small-bilingual"
    wandb_api_key: str = ""

    # GPU
    cuda_visible_devices: str = "1"

    # Online datasets
    online_train_datasets: List[dict] = field(default_factory=list)
    online_test_datasets: List[dict] = field(default_factory=list)

    # Local datasets
    local_train_datasets: List[str] = field(default_factory=lambda: [
        "/mnt/data/data_create/output/dataset/train_dataset_20260515_134507",
        "/mnt/data/data_create/huggingface_datasets/train/biblemms_vie",
        "/mnt/data/data_create/huggingface_datasets/train/infore2_audiobooks",
        "/mnt/data/data_create/huggingface_datasets/train/lsvsc_train",
        "/mnt/data/data_create/huggingface_datasets/train/vieneu-tts-140h-dataset",
        "/mnt/data/data_create/huggingface_datasets/train/viet_bud500",
        "/mnt/data/data_create/huggingface_datasets/train/vietnamese_voice",
        "/mnt/data/data_create/huggingface_datasets/train/vimedcss_standardized",
        "/mnt/data/data_create/huggingface_datasets/train/vlsp2020_vinai_100h",
        "/mnt/data/data_create/huggingface_datasets/train/viVoice",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech_asr/train.clean.100",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech_asr/train.clean.360",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech_asr/train.other.500",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech-alignments/train_other_500",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech-alignments/train_clean_360",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech-alignments/train_clean_100",
    ])
    local_test_datasets: List[str] = field(default_factory=lambda: [
        "/mnt/data/data_create/huggingface_datasets/test/viet_bud500_test",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech_asr/test.clean",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech-alignments/test_clean",
        "/mnt/data/data_create/huggingface_datasets/train/librispeech-alignments/test_other",
        "/mnt/data/data_create/huggingface_datasets/test/vimedcss_standardized",
        "/mnt/data/data_create/huggingface_datasets/test/w2wmovie_test",
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
