# Đánh giá checkpoint trên tập xác thực hoặc kiểm thử

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from src.config import SummarizationConfig
from src.data import load_dataset_from_files, preprocess_for_seq2seq
from src.evaluation_io import (
    export_predictions as _export_predictions,
    summarize_results,
)
from src.metrics import (
    build_compute_metrics,
    compute_rouge,
    tokenize_vietnamese_for_rouge,
)
from src.model import load_model, load_tokenizer, verify_adapter_base_dependency
from src.utils import save_json, setup_logger

logger = setup_logger(__name__)

__all__ = [
    "build_compute_metrics",
    "compute_rouge",
    "evaluate_checkpoint",
    "summarize_results",
    "tokenize_vietnamese_for_rouge",
]


# Đánh giá điểm kiểm tra

def evaluate_checkpoint(
    model_path: str | Path,
    config: SummarizationConfig,
    output_dir: Optional[str | Path] = None,
    export_predictions: bool = True,
    split: str = "validation",
    base_model_path: Optional[str | Path] = None,
) -> dict[str, float]:
    # Đánh giá điểm kiểm tra đầy đủ hoặc bộ điều hợp LoRA trên một tập con
    split = _validate_split(split, config)
    model_path = Path(model_path)
    output_dir = Path(output_dir or model_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Đang đánh giá checkpoint: {model_path} trên split={split}")

    tokenizer, model = _load_evaluation_model(
        model_path,
        config,
        base_model_path,
    )
    raw_dataset, evaluation_dataset = _load_evaluation_dataset(
        config,
        tokenizer,
        split,
    )
    trainer = _build_evaluation_trainer(
        config,
        output_dir,
        tokenizer,
        model,
        evaluation_dataset,
    )

    logger.info(f"Đang chạy đánh giá trên split={split}...")
    predict_output = trainer.predict(
        test_dataset=evaluation_dataset,
        metric_key_prefix=split,
    )

    metrics = predict_output.metrics
    _log_metrics(metrics)
    save_json(metrics, output_dir / f"{split}_metrics.json")

    if export_predictions:
        _export_predictions(
            predictions=predict_output.predictions,
            labels=predict_output.label_ids,
            tokenizer=tokenizer,
            dataset=raw_dataset,
            output_path=output_dir / f"predictions_{split}.jsonl",
        )

    return metrics


def _validate_split(split: str, config: SummarizationConfig) -> str:
    # Chuẩn hóa và kiểm tra tập dữ liệu cần đánh giá
    normalized_split = split.strip().casefold()
    if normalized_split not in {"validation", "test"}:
        raise ValueError(
            f"Split không hợp lệ: {normalized_split!r}. "
            "Chỉ hỗ trợ 'validation' hoặc 'test'."
        )
    if normalized_split == "test" and not config.data.test_file:
        raise ValueError(
            "Đã chọn split='test' nhưng config.data.test_file đang để trống."
        )
    return normalized_split


def _load_evaluation_model(
    model_path: Path,
    config: SummarizationConfig,
    base_model_path: Optional[str | Path],
) -> tuple[Any, Any]:
    # Nạp checkpoint đầy đủ hoặc ghép mô hình nền với adapter LoRA
    if not (model_path / "adapter_config.json").exists():
        if base_model_path is not None:
            logger.warning(
                "Bỏ qua base_model_path vì model_path là checkpoint đầy đủ, "
                "không phải LoRA adapter."
            )

        model_config = replace(config.model, name_or_path=str(model_path))
        tokenizer = load_tokenizer(model_config)
        model = load_model(model_config, tokenizer, config.generation)
        return tokenizer, model

    from peft import PeftModel

    base_model_reference = _resolve_adapter_base(config, base_model_path)
    base_model_config = replace(
        config.model,
        name_or_path=base_model_reference,
    )
    logger.info(
        "Phát hiện LoRA adapter, đang tải mô hình cơ sở + adapter: "
        f"base={base_model_reference}, adapter={model_path}"
    )
    verify_adapter_base_dependency(base_model_reference, model_path)
    tokenizer = load_tokenizer(base_model_config)
    base_model = load_model(base_model_config, tokenizer, config.generation)
    model = PeftModel.from_pretrained(base_model, str(model_path))
    return tokenizer, model


def _resolve_adapter_base(
    config: SummarizationConfig,
    base_model_path: Optional[str | Path],
) -> str:
    # Xác định checkpoint nền dùng để nạp adapter
    if base_model_path is not None:
        return str(base_model_path)

    logger.warning(
        "Không truyền base_model_path cho LoRA adapter; đang dùng "
        f"config.model.name_or_path={config.model.name_or_path!r}. "
        "Nên truyền rõ checkpoint Phase 1 để kết quả có thể tái lập."
    )
    return config.model.name_or_path


def _load_evaluation_dataset(
    config: SummarizationConfig,
    tokenizer: Any,
    split: str,
) -> tuple[Any, Any]:
    # Nạp đúng tập dữ liệu và mã hóa cho model
    datasets = load_dataset_from_files(
        train_file=None,
        valid_file=config.data.valid_file if split == "validation" else None,
        test_file=config.data.test_file if split == "test" else None,
    )
    tokenized_datasets = preprocess_for_seq2seq(
        datasets,
        tokenizer,
        config.data,
    )
    return datasets[split], tokenized_datasets[split]


def _build_evaluation_trainer(
    config: SummarizationConfig,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    evaluation_dataset: Any,
) -> Any:
    # Tạo Seq2SeqTrainer chỉ phục vụ đánh giá
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    evaluation_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=config.generation.max_length,
        fp16=False,
        # Không tự bật wandb và yêu cầu đăng nhập trong notebook
        report_to=[],
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    return Seq2SeqTrainer(
        model=model,
        args=evaluation_args,
        eval_dataset=evaluation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )


def _log_metrics(metrics: dict[str, Any]) -> None:
    # In các chỉ số số học theo thứ tự tên
    logger.info("Kết quả đánh giá:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
