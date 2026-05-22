import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)


def load_model_for_inference(model_path: str, device: str = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, processor, device


def transcribe(
    audio_input,
    model_path: str,
    language: str = None,
    task: str = "transcribe",
    return_timestamps: bool = False,
):
    model, processor, device = load_model_for_inference(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device.index if device.startswith("cuda") else -1,
    )

    generate_kwargs = {"task": task}
    if language:
        generate_kwargs["language"] = language

    result = pipe(
        audio_input,
        generate_kwargs=generate_kwargs,
        return_timestamps=return_timestamps,
    )

    return result


def batch_transcribe(audio_files: list, model_path: str, language: str = None, task: str = "transcribe"):
    results = []
    for audio in audio_files:
        result = transcribe(audio, model_path, language, task)
        results.append(result)
    return results
