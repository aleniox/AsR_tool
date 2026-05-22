import os
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
)
from config import TrainingConfig


def load_model_and_processor(config: TrainingConfig):
    print(f"Loading model and processor: {config.model_name_or_path}...")

    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name_or_path, dropout=0.1
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name_or_path)

    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_name_or_path, language=None, task=config.task
    )
    processor = WhisperProcessor.from_pretrained(
        config.model_name_or_path, language=None, task=config.task
    )

    return model, feature_extractor, tokenizer, processor


def add_extra_tokens(model, tokenizer, processor):
    extra_tokens = ["<|noise|>", "<|music|>", "<|silence|>", "<|laugh|>", "<|breath|>"]
    missing = [t for t in extra_tokens if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        processor.tokenizer = tokenizer
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, processor


def configure_multilingual(model, tokenizer, config: TrainingConfig):
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = None
    model.generation_config.task = config.task
    return model


def configure_anti_hallucination(model, tokenizer, config: TrainingConfig):
    model.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
    model.generation_config.condition_on_previous_text = config.condition_on_previous_text
    model.generation_config.predict_timestamps = config.predict_timestamps
    model.generation_config.compression_ratio_threshold = config.compression_ratio_threshold
    model.generation_config.logprob_threshold = config.logprob_threshold
    model.generation_config.no_speech_threshold = config.no_speech_threshold
    model.config.begin_suppress_tokens = [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    ]
    return model


def configure_language_suppression(model, tokenizer, allowed_languages=None):
    if allowed_languages is None:
        allowed_languages = ["vi", "en"]

    allowed_tokens = [
        tokenizer.convert_tokens_to_ids(f"<|{lang}|>") for lang in allowed_languages
    ]

    all_tokens = tokenizer.get_vocab()
    language_tokens = [
        id
        for token, id in all_tokens.items()
        if len(token) == 6
        and token.startswith("<|")
        and token.endswith("|>")
        and id not in allowed_tokens
    ]

    suppress_list = list(set(model.config.suppress_tokens + language_tokens))
    model.config.suppress_tokens = suppress_list
    model.generation_config.suppress_tokens = suppress_list
    return model


def setup_model(config: TrainingConfig):
    model, feature_extractor, tokenizer, processor = load_model_and_processor(config)
    model = add_extra_tokens(model, tokenizer, processor)[0]
    model = configure_multilingual(model, tokenizer, config)
    model = configure_anti_hallucination(model, tokenizer, config)
    model = configure_language_suppression(model, tokenizer, config.languages)
    return model, feature_extractor, tokenizer, processor
