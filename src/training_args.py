# Chuyển cấu hình dự án sang TrainingArguments của Transformers

from __future__ import annotations

from pathlib import Path

from transformers import Seq2SeqTrainingArguments

from src.config import SummarizationConfig
from src.utils import detect_precision


def build_training_args(
    config: SummarizationConfig,
) -> Seq2SeqTrainingArguments:
    # Tạo đối số huấn luyện từ cấu hình YAML đã kiểm tra
    training = config.training
    precision = (
        detect_precision()
        if training.precision == "auto"
        else training.precision
    )

    return Seq2SeqTrainingArguments(
        output_dir=training.output_dir,
        seed=training.seed,

        # Tối ưu hóa
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        optim=training.optim,

        # Chia lô dữ liệu
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,

        # Độ chính xác hỗn hợp
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),

        # Đánh giá và điểm kiểm tra
        eval_strategy=training.eval_strategy,
        eval_steps=training.eval_steps,
        save_strategy=training.save_strategy,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=training.metric_for_best_model,
        greater_is_better=training.greater_is_better,

        # ROUGE cần chuỗi đã sinh thay vì chỉ giá trị logits
        predict_with_generate=True,
        generation_max_length=config.generation.max_length,

        # Ghi nhật ký
        label_smoothing_factor=training.label_smoothing_factor,
        report_to=["tensorboard"],
        logging_dir=str(Path(training.output_dir) / "logs"),
        logging_steps=training.logging_steps,
        ddp_find_unused_parameters=training.ddp_find_unused_parameters,
    )
