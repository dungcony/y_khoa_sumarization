# Kiểm tra an toàn và quản lý manifest khi huấn luyện LoRA

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import SummarizationConfig, config_to_dict
from src.model import fingerprint_full_checkpoint
from src.utils import (
    count_parameters,
    format_number,
    load_json,
    save_json,
    setup_logger,
)

logger = setup_logger(__name__)


def prepare_base_fingerprint(
    config: SummarizationConfig,
    output_dir: Path,
) -> dict[str, Any] | None:
    # Ghi nhận mô hình nền trước khi huấn luyện LoRA
    if not config.lora.enabled:
        return None

    base_fingerprint = fingerprint_full_checkpoint(config.model.name_or_path)
    if base_fingerprint is None:
        logger.warning(
            "Không tạo được fingerprint cho base model (có thể là Hub ID). "
            "Adapter manifest sẽ không thể xác minh tuyệt đối dependency."
        )
    if config.training.resume_from_checkpoint:
        verify_resume_manifest(config, output_dir, base_fingerprint)

    return base_fingerprint


def build_adapter_manifest(
    config: SummarizationConfig,
    adapter_path: Path,
    base_fingerprint: dict[str, Any] | None,
    parameter_stats: dict[str, int | float],
) -> dict[str, Any]:
    # Tạo tệp kê khai cho bộ điều hợp LoRA
    return {
        "artifact_type": "peft_lora_adapter",
        "phase": config.phase.name,
        "base_model_dependency": {
            "name_or_path": config.model.name_or_path,
            "required_for_loading": True,
            "fingerprint": base_fingerprint,
        },
        "adapter_path": str(adapter_path),
        "merged_into_base_model": False,
        "lora": config_to_dict(config)["lora"],
        "parameters": parameter_stats,
    }


def verify_resume_manifest(
    config: SummarizationConfig,
    output_dir: Path,
    current_base_fingerprint: dict[str, Any] | None,
) -> None:
    # Chặn tiếp tục LoRA bằng nhầm mô hình nền hoặc cấu hình bộ điều hợp
    resume_path = Path(config.training.resume_from_checkpoint or "").resolve()
    if output_dir.resolve() not in resume_path.parents:
        raise ValueError(
            "LoRA resume checkpoint phải nằm trong đúng training.output_dir."
        )
    if not (resume_path / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"LoRA resume checkpoint thiếu trainer_state.json: {resume_path}"
        )

    manifest_path = output_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Không thể resume LoRA an toàn vì run cũ thiếu "
            f"adapter_manifest.json: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    expected_fingerprint = manifest.get("base_model_dependency", {}).get(
        "fingerprint"
    )
    if expected_fingerprint is None or current_base_fingerprint is None:
        raise ValueError(
            "Không thể resume LoRA an toàn vì base fingerprint bị thiếu."
        )
    if expected_fingerprint != current_base_fingerprint:
        raise ValueError(
            "Checkpoint resume thuộc một Phase 1 base khác; fingerprint "
            "không khớp base hiện tại."
        )

    current_lora_config = config_to_dict(config)["lora"]
    if manifest.get("lora") != current_lora_config:
        raise ValueError(
            "Cấu hình LoRA hiện tại khác run cần resume; không được đổi "
            "rank/alpha/dropout/target_modules giữa run."
        )
    if manifest.get("phase") != config.phase.name:
        raise ValueError("Phase hiện tại khác phase trong adapter manifest.")

    logger.info("Xác minh manifest resume LoRA thành công")


def verify_trainable_parameters(model: Any) -> dict[str, int | float]:
    # Kiểm tra LoRA không vô tình mở khóa trọng số mô hình nền
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "LoRA đã bật nhưng không có tham số nào có thể huấn luyện. "
            "Hãy kiểm tra lora.target_modules."
        )

    unexpected_trainable = [
        name for name, _ in trainable_parameters if "lora_" not in name
    ]
    if unexpected_trainable:
        preview = ", ".join(unexpected_trainable[:10])
        suffix = " ..." if len(unexpected_trainable) > 10 else ""
        raise RuntimeError(
            "Phát hiện trọng số không thuộc LoRA vẫn có thể huấn luyện; "
            f"dừng Phase 2 để bảo vệ checkpoint Phase 1: {preview}{suffix}"
        )

    parameter_counts = count_parameters(model)
    trainable_tensor_count = len(trainable_parameters)
    logger.info(
        "Kiểm tra an toàn LoRA đạt: chỉ %s tensor LoRA (%s tham số, %.4f%%) "
        "được huấn luyện; %s tham số base đã được đóng băng.",
        trainable_tensor_count,
        format_number(parameter_counts["trainable"]),
        100 * parameter_counts["trainable"] / parameter_counts["total"],
        format_number(parameter_counts["frozen"]),
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return {
        "total": parameter_counts["total"],
        "trainable": parameter_counts["trainable"],
        "frozen": parameter_counts["frozen"],
        "trainable_percent": parameter_counts["trainable_percent"],
        "trainable_tensor_count": trainable_tensor_count,
    }


def verify_base_unchanged(
    config: SummarizationConfig,
    expected_fingerprint: dict[str, Any] | None,
) -> None:
    # Bảo đảm checkpoint nền không đổi trong lúc huấn luyện adapter
    if expected_fingerprint is None:
        return

    current_fingerprint = fingerprint_full_checkpoint(config.model.name_or_path)
    if current_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Checkpoint base Phase 1 đã thay đổi trong lúc train LoRA; "
            "dừng trước khi xuất adapter."
        )


def save_adapter_manifest(
    manifest: dict[str, Any],
    *paths: Path,
) -> None:
    # Lưu cùng manifest tại các vị trí cần thiết
    for path in paths:
        save_json(manifest, path)
