from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch

from src.config import (
    GenerationConfig,
    ModelConfig,
    SummarizationConfig,
)
from src.data import clean_text
from src.model import load_model_for_inference
from src.utils import setup_logger

logger = setup_logger(__name__)


# Suy luận cho một văn bản

def _resolve_base_model_path(
    model_path: str | Path | None,
    base_model_path: str | Path | None,
    config: Optional[SummarizationConfig],
) -> str:
    # Gộp hai tên tham số dùng cho điểm kiểm tra nền
    if model_path is not None and base_model_path is not None:
        raise ValueError(
            "Chỉ truyền một trong hai tham số model_path hoặc "
            "base_model_path; hai tham số này là alias của nhau."
        )

    resolved = base_model_path if base_model_path is not None else model_path
    if resolved is None and config is not None:
        resolved = config.model.name_or_path

    if resolved is None or not str(resolved).strip():
        raise ValueError(
            "Thiếu full base checkpoint. Hãy truyền model_path hoặc "
            "base_model_path (ví dụ checkpoint Phase 1)."
        )

    return str(resolved)


def _resolve_generation_config(
    config: Optional[SummarizationConfig],
    *,
    num_beams: int | None,
    max_length: int | None,
    min_length: int | None,
    repetition_penalty: float | None,
    length_penalty: float | None,
    no_repeat_ngram_size: int | None,
) -> GenerationConfig:
    # Tạo cấu hình sinh văn bản mà không sửa đối tượng của bên gọi
    if config is None:
        resolved = GenerationConfig(
            num_beams=4,
            max_length=200,
            min_length=30,
            repetition_penalty=1.05,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
    else:
        resolved = replace(config.generation)

    overrides = {
        "num_beams": num_beams,
        "max_length": max_length,
        "min_length": min_length,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    explicit_overrides = {
        name: value
        for name, value in overrides.items()
        if value is not None
    }
    if explicit_overrides:
        resolved = replace(resolved, **explicit_overrides)

    return resolved


def _resolve_source_settings(
    config: Optional[SummarizationConfig],
    source_prefix: str | None,
    max_source_length: int | None,
) -> tuple[str, int]:
    # Lấy tiền tố và giới hạn đầu vào từ tham số hoặc cấu hình
    resolved_prefix = (
        source_prefix
        if source_prefix is not None
        else (config.data.source_prefix if config is not None else "summarize: ")
    )
    resolved_max_length = (
        max_source_length
        if max_source_length is not None
        else (config.data.max_source_length if config is not None else 768)
    )
    return resolved_prefix, resolved_max_length


def _build_model_config(
    base_model_path: str,
    config: Optional[SummarizationConfig],
) -> ModelConfig:
    # Tạo cấu hình model mà không sửa cấu hình của bên gọi
    if config is not None:
        return replace(config.model, name_or_path=base_model_path)

    model_config = ModelConfig(name_or_path=base_model_path)
    if any(name in base_model_path.lower() for name in ("t5", "mt5", "vit5")):
        model_config.use_fast_tokenizer = False
    return model_config


def _load_inference_runtime(
    model_config: ModelConfig,
    generation_config: GenerationConfig,
    adapter_path: str | Path | None,
) -> tuple[Any, Any, torch.device]:
    # Nạp model một lần và chuyển sang thiết bị suy luận
    tokenizer, model = load_model_for_inference(
        model_config,
        generation_config,
        adapter_path=adapter_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _generate_summaries(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    generation_config: GenerationConfig,
    source_prefix: str,
    max_source_length: int,
    batch_size: int,
) -> list[str]:
    # Làm sạch, mã hóa và sinh tóm tắt theo từng lô
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn 0")

    summaries = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        input_texts = [
            source_prefix + clean_text(text)
            for text in batch_texts
        ]
        inputs = tokenizer(
            input_texts,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=generation_config.max_length,
                min_length=generation_config.min_length,
                num_beams=generation_config.num_beams,
                length_penalty=generation_config.length_penalty,
                no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                repetition_penalty=generation_config.repetition_penalty,
                early_stopping=generation_config.early_stopping,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(summary.strip() for summary in decoded)
        logger.info(
            f"Đã xử lý {min(start + batch_size, len(texts))}/"
            f"{len(texts)} văn bản"
        )

    return summaries


def summarize(
    text: str,
    model_path: str | Path | None = None,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str | None = None,
    max_source_length: int | None = None,
    num_beams: int | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
    repetition_penalty: float | None = None,
    length_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    adapter_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> str:
    # Sinh tóm tắt bằng điểm kiểm tra đầy đủ hoặc LoRA
    resolved_base_path = _resolve_base_model_path(
        model_path,
        base_model_path,
        config,
    )
    resolved_prefix, resolved_source_length = _resolve_source_settings(
        config,
        source_prefix,
        max_source_length,
    )
    resolved_generation = _resolve_generation_config(
        config,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    model_config = _build_model_config(resolved_base_path, config)
    tokenizer, model, device = _load_inference_runtime(
        model_config,
        resolved_generation,
        adapter_path,
    )
    return _generate_summaries(
        texts=[text],
        tokenizer=tokenizer,
        model=model,
        device=device,
        generation_config=resolved_generation,
        source_prefix=resolved_prefix,
        max_source_length=resolved_source_length,
        batch_size=1,
    )[0]


# Suy luận theo lô

def summarize_batch(
    texts: list[str],
    model_path: str | Path | None = None,
    config: Optional[SummarizationConfig] = None,
    source_prefix: str | None = None,
    max_source_length: int | None = None,
    batch_size: int = 8,
    adapter_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> list[str]:
    # Sinh theo lô để chỉ phải nạp mô hình một lần
    resolved_base_path = _resolve_base_model_path(
        model_path,
        base_model_path,
        config,
    )
    resolved_prefix, resolved_source_length = _resolve_source_settings(
        config,
        source_prefix,
        max_source_length,
    )
    generation_config = (
        replace(config.generation)
        if config is not None
        else GenerationConfig()
    )
    model_config = _build_model_config(resolved_base_path, config)
    tokenizer, model, device = _load_inference_runtime(
        model_config,
        generation_config,
        adapter_path,
    )
    return _generate_summaries(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        generation_config=generation_config,
        source_prefix=resolved_prefix,
        max_source_length=resolved_source_length,
        batch_size=batch_size,
    )


def main() -> None:
    # Giữ điểm vào cũ cho scripts và python -m src.predict
    from src.prediction_cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
