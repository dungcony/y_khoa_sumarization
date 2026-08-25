from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
)

from src.config import SummarizationConfig, config_to_dict
from src.data import load_and_preprocess
from src.lora_training import (
    build_adapter_manifest,
    prepare_base_fingerprint,
    save_adapter_manifest,
    verify_base_unchanged,
    verify_trainable_parameters,
)
from src.metrics import build_compute_metrics
from src.model import (
    apply_lora,
    enable_gradient_checkpointing,
    freeze_encoder,
    load_model,
    load_tokenizer,
)
from src.training_args import build_training_args
from src.utils import (
    format_duration,
    get_device_info,
    save_json,
    set_seed,
    setup_logger,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _PreparedModel:
    # Các thành phần cần dùng trong suốt quá trình huấn luyện

    tokenizer: Any
    model: Any
    base_fingerprint: dict[str, Any] | None
    adapter_manifest: dict[str, Any] | None


def train(config: SummarizationConfig) -> dict[str, float]:
    # Chạy huấn luyện và đánh giá trên tập xác thực
    start_time = time.time()
    tc = config.training

    _validate_training_setup(config)
    logger.info("Bắt đầu huấn luyện")

    output_dir = _prepare_run(config)
    prepared_model = _prepare_model(config, output_dir)
    datasets = _load_training_datasets(config, prepared_model.tokenizer)
    trainer = _build_trainer(config, prepared_model, datasets)

    logger.info(f"Đang train (epochs={tc.num_train_epochs}, lr={tc.learning_rate})")
    train_result = trainer.train(resume_from_checkpoint=tc.resume_from_checkpoint)

    _save_best_checkpoint(config, output_dir, prepared_model, trainer)

    logger.info("Đang đánh giá trên validation split...")
    eval_results = trainer.evaluate(metric_key_prefix="eval")
    _save_training_results(config, output_dir, train_result.metrics, eval_results)

    logger.info(f"Hoàn thành sau {format_duration(time.time() - start_time)}")
    return eval_results


def _validate_training_setup(config: SummarizationConfig) -> None:
    # Kiểm tra các tùy chọn không được phép kết hợp
    if config.lora.enabled and config.training.freeze_encoder:
        raise ValueError(
            "Không kết hợp training.freeze_encoder=true với LoRA. PEFT đã "
            "đóng băng toàn bộ base model; hãy đặt freeze_encoder=false."
        )


def _prepare_run(config: SummarizationConfig) -> Path:
    # Chuẩn bị thiết bị, thư mục đầu ra và hạt giống
    device = get_device_info()
    logger.info(
        f"Thiết bị: {device['device'].upper()} | Số lượng: {device['num_gpus']}"
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.training.seed)
    return output_dir


def _prepare_model(
    config: SummarizationConfig,
    output_dir: Path,
) -> _PreparedModel:
    # Tải model, áp dụng chiến lược huấn luyện và chuẩn bị thông tin LoRA
    base_fingerprint = prepare_base_fingerprint(config, output_dir)

    logger.info("Đang tải model và tokenizer...")
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, tokenizer, config.generation)

    if config.training.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if config.training.freeze_encoder:
        freeze_encoder(model)
    model = apply_lora(model, config.lora)

    adapter_manifest = None
    if config.lora.enabled:
        parameter_stats = verify_trainable_parameters(model)
        adapter_manifest = build_adapter_manifest(
            config,
            output_dir / "best",
            base_fingerprint,
            parameter_stats,
        )
        # Ghi sớm để kiểm tra mô hình nền khi tiếp tục huấn luyện
        save_adapter_manifest(
            adapter_manifest,
            output_dir / "adapter_manifest.json",
        )

    return _PreparedModel(
        tokenizer=tokenizer,
        model=model,
        base_fingerprint=base_fingerprint,
        adapter_manifest=adapter_manifest,
    )


def _load_training_datasets(
    config: SummarizationConfig,
    tokenizer: Any,
) -> Any:
    # Chỉ nạp train và validation để tập test không tham gia chọn model
    logger.info("Đang tiền xử lý dữ liệu...")
    training_data_config = replace(config.data, test_file="")
    return load_and_preprocess(tokenizer, training_data_config)


def _build_trainer(
    config: SummarizationConfig,
    prepared_model: _PreparedModel,
    datasets: Any,
) -> Seq2SeqTrainer:
    # Ghép model, dữ liệu và cấu hình thành bộ huấn luyện
    tokenizer = prepared_model.tokenizer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=prepared_model.model,
        padding=True,
        label_pad_token_id=-100,
    )

    callbacks = []
    patience = config.training.early_stopping_patience
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    return Seq2SeqTrainer(
        model=prepared_model.model,
        args=build_training_args(config),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )


def _save_best_checkpoint(
    config: SummarizationConfig,
    output_dir: Path,
    prepared_model: _PreparedModel,
    trainer: Seq2SeqTrainer,
) -> None:
    # Kiểm tra mô hình nền rồi lưu checkpoint tốt nhất
    logger.info("Đang lưu best checkpoint...")
    best_dir = output_dir / "best"

    verify_base_unchanged(config, prepared_model.base_fingerprint)

    trainer.save_model(str(best_dir))
    prepared_model.tokenizer.save_pretrained(str(best_dir))

    if prepared_model.adapter_manifest is not None:
        # Lưu cùng adapter để mô tả quan hệ với mô hình nền
        save_adapter_manifest(
            prepared_model.adapter_manifest,
            best_dir / "adapter_manifest.json",
            output_dir / "adapter_manifest.json",
        )


def _save_training_results(
    config: SummarizationConfig,
    output_dir: Path,
    train_metrics: dict[str, Any],
    eval_results: dict[str, float],
) -> None:
    # Lưu chỉ số huấn luyện, đánh giá và cấu hình đã dùng
    train_metrics["train_runtime_formatted"] = format_duration(
        train_metrics.get("train_runtime", 0)
    )
    save_json(train_metrics, output_dir / "train_results.json")
    save_json(eval_results, output_dir / "eval_results.json")
    save_json(config_to_dict(config), output_dir / "resolved_config.json")

