# Nạp bộ mã hóa, mô hình chuỗi sang chuỗi và bộ điều hợp LoRA

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

from src.config import GenerationConfig as GenConfigDC
from src.config import LoraConfig, ModelConfig
from src.utils import count_parameters, format_number, setup_logger

logger = setup_logger(__name__)


# Nạp bộ mã hóa

def load_tokenizer(model_config: ModelConfig) -> Any:
    # T5/ViT5 mặc định dùng SentencePiece bản chậm
    model_name = model_config.name_or_path
    use_fast = model_config.use_fast_tokenizer

    # SentencePiece bản chậm ổn định hơn với các điểm kiểm tra T5/ViT5 hiện dùng
    is_t5_model = _is_t5_family(model_name)
    if is_t5_model and not use_fast:
        logger.info(f"Đang tải T5 SentencePiece tokenizer cho: {model_name}")
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                legacy=True,
                cache_dir=model_config.cache_dir,
            )
        except Exception as e:
            logger.warning(
                f"T5Tokenizer thất bại ({e}), chuyển về AutoTokenizer"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                cache_dir=model_config.cache_dir,
            )
    else:
        logger.info(f"Đang tải tokenizer cho: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
        )

    logger.info(
        f"Đã tải tokenizer: vocab_size={tokenizer.vocab_size}, "
        f"type={type(tokenizer).__name__}"
    )
    return tokenizer


# Nạp mô hình

def load_model(
    model_config: ModelConfig,
    tokenizer: Any,
    generation_config: GenConfigDC | None = None,
) -> Any:
    # Nạp mô hình chuỗi sang chuỗi và áp dụng cấu hình sinh văn bản nếu có
    model_name = model_config.name_or_path
    logger.info(f"Đang tải mô hình: {model_name}")

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
    )

    if model_config.dropout is not None:
        _set_dropout(config, model_config.dropout)
        logger.info(f"Đã ghi đè dropout thành: {model_config.dropout}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        logger.info("Đã đặt pad_token thành eos_token")

    model.resize_token_embeddings(len(tokenizer))

    if generation_config:
        gen_cfg = getattr(model, "generation_config", GenerationConfig())
        gen_cfg.max_length = generation_config.max_length
        gen_cfg.max_new_tokens = generation_config.max_new_tokens
        gen_cfg.min_length = generation_config.min_length
        gen_cfg.num_beams = generation_config.num_beams
        gen_cfg.length_penalty = generation_config.length_penalty
        gen_cfg.no_repeat_ngram_size = generation_config.no_repeat_ngram_size
        gen_cfg.repetition_penalty = generation_config.repetition_penalty
        gen_cfg.do_sample = generation_config.do_sample
        gen_cfg.early_stopping = generation_config.early_stopping
        
        # GenerationConfig mới không tự kế thừa mã bắt đầu của bộ giải mã
        if gen_cfg.decoder_start_token_id is None:
            gen_cfg.decoder_start_token_id = getattr(model.config, "decoder_start_token_id", tokenizer.pad_token_id)
            
        model.generation_config = gen_cfg

    params = count_parameters(model)
    logger.info(
        f"Đã tải mô hình: {format_number(params['total'])} tổng số tham số, "
        f"{format_number(params['trainable'])} có thể huấn luyện "
        f"({params['trainable_percent']}%)"
    )

    if params["total"] > model_config.max_parameters:
        raise ValueError(
            f"Mô hình có {format_number(params['total'])} tham số, "
            f"vượt quá giới hạn {format_number(model_config.max_parameters)}"
        )

    return model


def is_lora_adapter_checkpoint(path: str | Path) -> bool:
    # Nhận diện bộ điều hợp LoRA qua adapter_config.json
    checkpoint_path = Path(path).expanduser()
    return (
        checkpoint_path.is_dir()
        and (checkpoint_path / "adapter_config.json").is_file()
    )


def fingerprint_full_checkpoint(
    path: str | Path,
) -> dict[str, Any] | None:
    # Băm điểm kiểm tra cục bộ để xác nhận đúng mô hình nền
    checkpoint_dir = Path(path).expanduser()
    if not checkpoint_dir.is_dir():
        return None

    config_path = checkpoint_dir / "config.json"
    weight_paths = sorted({
        *checkpoint_dir.glob("model*.safetensors"),
        *checkpoint_dir.glob("pytorch_model*.bin"),
    })
    weight_paths = [item for item in weight_paths if item.is_file()]
    if not config_path.is_file() or not weight_paths:
        return None

    tokenizer_paths = [
        checkpoint_dir / name
        for name in (
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if (checkpoint_dir / name).is_file()
    ]
    files = [config_path, *tokenizer_paths, *weight_paths]
    digest = hashlib.sha256()
    records: list[dict[str, str | int]] = []
    for file_path in files:
        relative_name = file_path.relative_to(checkpoint_dir).as_posix()
        size = file_path.stat().st_size
        records.append({"name": relative_name, "size": size})
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "files": records,
    }


def verify_adapter_base_dependency(
    base_model_path: str | Path,
    adapter_path: str | Path,
) -> bool:
    # Đối chiếu dấu vân tay giữa bộ điều hợp LoRA và điểm kiểm tra nền
    adapter_dir = Path(adapter_path).expanduser()
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        logger.warning(
            "Adapter không có adapter_manifest.json; không thể xác minh đúng "
            "checkpoint base. Chỉ dùng artifact này nếu provenance đã được "
            "kiểm tra bằng cách khác."
        )
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Không đọc được adapter manifest '{manifest_path}': {exc}"
        ) from exc

    dependency = manifest.get("base_model_dependency", {})
    expected = dependency.get("fingerprint")
    if not isinstance(expected, dict) or not expected.get("digest"):
        logger.warning(
            "Adapter manifest không có base fingerprint; không thể xác minh "
            "đúng cặp base-adapter."
        )
        return False

    actual = fingerprint_full_checkpoint(base_model_path)
    if actual is None:
        raise ValueError(
            "Adapter có base fingerprint nhưng base_model_path không phải "
            "full checkpoint local có thể xác minh. Hãy truyền đúng thư mục "
            "Phase 1 best/, không dùng Hub ID hoặc adapter path."
        )

    if (
        expected.get("algorithm") != actual["algorithm"]
        or expected.get("digest") != actual["digest"]
    ):
        raise ValueError(
            "LoRA adapter không thuộc checkpoint base đã chọn: fingerprint "
            f"mong đợi={expected.get('digest')}, thực tế={actual['digest']}."
        )

    logger.info("Xác minh fingerprint base-adapter thành công")
    return True


def load_model_for_inference(
    model_config: ModelConfig,
    generation_config: GenConfigDC | None = None,
    adapter_path: str | Path | None = None,
) -> tuple[Any, Any]:
    # Nạp điểm kiểm tra đầy đủ và gắn bộ điều hợp LoRA mà không hợp nhất trọng số
    base_model_path = model_config.name_or_path
    if is_lora_adapter_checkpoint(base_model_path):
        raise ValueError(
            f"Đã phát hiện LoRA adapter tại '{base_model_path}', nhưng tham số "
            "model/base phải trỏ tới full checkpoint. Hãy truyền checkpoint "
            "Phase 1 làm base và truyền thư mục này qua adapter_path."
        )

    adapter_dir: Path | None = None
    if adapter_path is not None:
        # Kiểm tra quan hệ phụ thuộc trước khi nạp mô hình nền lớn
        adapter_dir = Path(adapter_path).expanduser()
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục LoRA adapter: '{adapter_dir}'."
            )
        if not adapter_dir.is_dir():
            raise ValueError(
                f"LoRA adapter phải là một thư mục, nhưng nhận được: "
                f"'{adapter_dir}'."
            )

        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError(
                f"'{adapter_dir}' không phải LoRA adapter hợp lệ: thiếu "
                "adapter_config.json. Nếu đây là full checkpoint, hãy truyền nó "
                "qua model_path/base_model_path và bỏ adapter_path."
            )

        verify_adapter_base_dependency(base_model_path, adapter_dir)

    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config, tokenizer, generation_config)

    if adapter_dir is None:
        model.eval()
        return tokenizer, model

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Cần cài package 'peft' để nạp LoRA adapter."
        ) from exc

    logger.info(
        f"Đã phát hiện LoRA adapter: {adapter_dir}; "
        f"đang gắn vào base model: {base_model_path}"
    )
    try:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            is_trainable=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Không thể nạp LoRA adapter '{adapter_dir}' vào base model "
            f"'{base_model_path}'. Hãy kiểm tra adapter đã được train từ "
            f"đúng checkpoint Phase 1. Chi tiết: {exc}"
        ) from exc

    model.eval()
    logger.info("Nạp LoRA adapter thành công (không merge vào base model)")
    return tokenizer, model


# LoRA

def apply_lora(model: Any, lora_config: LoraConfig) -> Any:
    # Gắn LoRA vào các phép chiếu đã cấu hình
    if not lora_config.enabled:
        logger.info("LoRA bị vô hiệu hóa, sử dụng fine-tuning toàn bộ (full fine-tuning)")
        return model

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    target_modules = _get_lora_target_modules(
        model, lora_config.target_modules
    )

    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)

    params = count_parameters(model)
    logger.info(
        f"Đã áp dụng LoRA (rank={lora_config.r}): "
        f"{format_number(params['trainable'])} tham số có thể huấn luyện "
        f"({params['trainable_percent']}% trong số {format_number(params['total'])})"
    )

    return model


# Cấu hình mô hình

def enable_gradient_checkpointing(model: Any) -> None:
    # Bật lưu điểm kiểm tra gradient nếu mô hình hỗ trợ
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Đã kích hoạt gradient checkpointing")
    else:
        logger.warning("Mô hình không hỗ trợ gradient checkpointing")


def freeze_encoder(model: Any) -> None:
    # Đóng băng bộ mã hóa, chỉ huấn luyện bộ giải mã và chú ý chéo
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen = sum(
            p.numel() for p in model.encoder.parameters()
        )
        logger.info(f"Đã đóng băng bộ mã hóa: {format_number(frozen)} tham số")
    else:
        logger.warning("Mô hình không có thuộc tính 'encoder' để đóng băng")


def _is_t5_family(model_name: str) -> bool:
    # Nhận diện T5, ViT5 và mT5
    name_lower = model_name.lower()
    return any(keyword in name_lower for keyword in ["t5", "mt5"])


def _set_dropout(config: Any, dropout: float) -> None:
    # Tên trường dropout khác nhau tùy kiến trúc mô hình
    if hasattr(config, "dropout_rate"):
        config.dropout_rate = dropout
    if hasattr(config, "dropout"):
        config.dropout = dropout
    if hasattr(config, "attention_dropout"):
        config.attention_dropout = dropout
    if hasattr(config, "activation_dropout"):
        config.activation_dropout = dropout


def _get_lora_target_modules(
    model: Any,
    target_modules: str,
) -> list[str]:
    # Chọn phép chiếu LoRA theo kiến trúc mô hình
    if target_modules != "auto":
        return [m.strip() for m in target_modules.split(",")]

    model_type = getattr(model.config, "model_type", "").lower()

    if model_type in ("t5", "mt5"):
        modules = ["q", "v"]
    elif "bart" in model_type:
        modules = ["q_proj", "v_proj"]
    else:
        # Phần lớn Transformer chuỗi sang chuỗi dùng tên phép chiếu này
        modules = ["q_proj", "v_proj"]
        logger.warning(
            f"Loại mô hình không xác định '{model_type}', dùng mặc định là {modules}"
        )

    logger.info(f"Các module mục tiêu của LoRA (tự động): {modules}")
    return modules
