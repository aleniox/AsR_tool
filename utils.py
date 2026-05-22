import re
import os
import gc
import warnings
import torch
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'(\d)\.(\d)', r'\1TOKEN_P\2', text)
    text = re.sub(r'(\d),(\d)', r'\1TOKEN_C\2', text)
    text = re.sub(r'(\d):(\d)', r'\1TOKEN_COL\2', text)
    text = re.sub(r'[.,?!:;\'\"()\[\]{}—–-]', '', text)
    text = text.replace('TOKEN_P', '.')
    text = text.replace('TOKEN_C', ',')
    text = text.replace('TOKEN_COL', ':')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_language_from_path(path_or_config) -> str:
    path_str = str(path_or_config).lower()
    vi_keywords = ['vi', 'viet', 'vietnamese', 'bud500', 'infore', 'vlsp', 'vivoice', 'vieneu', 'vimedcss']
    en_keywords = ['en', 'english', 'librispeech', 'libritts', 'commonvoice']
    for keyword in vi_keywords:
        if keyword in path_str:
            return "vi"
    for keyword in en_keywords:
        if keyword in path_str:
            return "en"
    return "vi"


def is_audio_valid(batch) -> list:
    results = []
    for audio_item in batch["audio"]:
        try:
            if audio_item is None:
                results.append(False)
                continue
            if isinstance(audio_item, dict) and audio_item.get("path") is not None:
                path = audio_item["path"]
                if not os.path.exists(path):
                    results.append(False)
                    continue
                if os.path.getsize(path) < 1024:
                    results.append(False)
                    continue
            elif isinstance(audio_item, dict) and audio_item.get("bytes") is not None:
                if len(audio_item["bytes"]) < 1024:
                    results.append(False)
                    continue
            else:
                results.append(False)
                continue
            results.append(True)
        except Exception:
            results.append(False)
    return results


def show_lang_stats(dataset, name="Dataset") -> dict:
    languages = dataset["language"]
    vi_count = languages.count("vi")
    en_count = languages.count("en")
    total = len(languages)
    stats = {
        "name": name,
        "vi": vi_count,
        "en": en_count,
        "total": total,
        "vi_pct": vi_count / total * 100 if total > 0 else 0,
        "en_pct": en_count / total * 100 if total > 0 else 0,
    }
    return stats


def show_lang_stats_str(dataset, name="Dataset") -> str:
    stats = show_lang_stats(dataset, name)
    lines = [
        f"Thống kê ngôn ngữ cho {stats['name']}:",
        f"  - Tiếng Việt (vi): {stats['vi']} mẫu ({stats['vi_pct']:.1f}%)",
        f"  - Tiếng Anh (en):  {stats['en']} mẫu ({stats['en_pct']:.1f}%)",
        f"  - Tổng cộng:       {stats['total']} mẫu",
    ]
    return "\n".join(lines)


def get_num_cpus() -> int:
    import multiprocessing
    try:
        if os.path.exists("/.dockerenv"):
            return min(4, multiprocessing.cpu_count())
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
