# Đọc và kiểm tra cấu hình YAML của quy trình

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml


# Cấu trúc cấu hình

@dataclass
class PhaseConfig:
    # Thông tin của một giai đoạn huấn luyện

    name: str = "default"
    description: str = ""


@dataclass
class ModelConfig:
    # Cấu hình mô hình và bộ mã hóa

    name_or_path: str = "VietAI/vit5-base"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    max_parameters: int = 3_000_000_000
    dropout: Optional[float] = None


@dataclass
class DataConfig:
    # Đường dẫn dữ liệu và giới hạn mã từ

    train_file: str = ""
    valid_file: str = ""
    test_file: str = ""
    source_prefix: str = "summarize: "
    max_source_length: int = 768
    max_target_length: int = 160
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    # Siêu tham số huấn luyện và điểm kiểm tra

    output_dir: str = "outputs/default"
    seed: int = 42

    # Lịch huấn luyện
    num_train_epochs: int = 3
    max_steps: int = -1

    # Chia lô dữ liệu
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2

    # Bộ tối ưu
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"

    # Điều chuẩn
    label_smoothing_factor: float = 0.05

    # Độ chính xác số học
    precision: str = "auto"

    # Kiểm soát bộ nhớ
    gradient_checkpointing: bool = False
    freeze_encoder: bool = False

    # Đánh giá và điểm kiểm tra
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 100

    # Chọn mô hình
    metric_for_best_model: str = "rougeL"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True

    # Dừng sớm
    early_stopping_patience: int = 5

    # Tiếp tục lần huấn luyện trước
    resume_from_checkpoint: Optional[str] = None

    # Huấn luyện phân tán
    ddp_find_unused_parameters: Optional[bool] = None


@dataclass
class GenerationConfig:
    # Tham số giải mã

    max_length: int = 200
    max_new_tokens: Optional[int] = None
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.0
    do_sample: bool = False
    early_stopping: bool = True


@dataclass
class LoraConfig:
    # Cấu hình PEFT/LoRA

    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "auto"


@dataclass
class SummarizationConfig:
    # Cấu hình đầy đủ của một lần chạy

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)


# Nạp và kiểm tra cấu hình

def load_config(config_path: str | Path) -> SummarizationConfig:
    # Đọc YAML, điền giá trị mặc định và kiểm tra cấu hình
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            raw = {} if loaded is None else loaded
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML không hợp lệ trong '{config_path}': {exc}") from exc

    return _build_config(raw)


def apply_overrides(
    config: SummarizationConfig,
    overrides: dict[str, Any],
) -> SummarizationConfig:
    # Áp dụng giá trị ghi đè từ CLI dạng section.field trên bản sao cấu hình
    config = copy.deepcopy(config)
    valid_sections = {item.name for item in fields(SummarizationConfig)}

    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Khóa ghi đè phải ở định dạng 'section.field', nhận được: '{key}'"
            )

        section_name, field_name = parts
        if section_name not in valid_sections:
            raise ValueError(
                f"Không rõ phần cấu hình: '{section_name}'. "
                f"Các phần hợp lệ: phase, model, data, training, generation, lora"
            )
        section = getattr(config, section_name)

        valid_fields = {item.name for item in fields(type(section))}
        if field_name not in valid_fields:
            raise ValueError(
                f"Không rõ trường '{field_name}' trong phần '{section_name}'"
            )

        # Giá trị ghi đè từ CLI thường ở dạng chuỗi
        current_value = getattr(section, field_name)
        expected_type = get_type_hints(type(section)).get(field_name)
        converted_value = _convert_type(
            value,
            current_value,
            f"{section_name}.{field_name}",
            expected_type,
        )
        setattr(section, field_name, converted_value)

    return validate_config(config)


def config_to_dict(config: SummarizationConfig) -> dict[str, Any]:
    # Chuyển cấu hình thành từ điển để ghi nhật ký hoặc lưu tệp
    from dataclasses import asdict
    return asdict(config)


def validate_config(config: SummarizationConfig) -> SummarizationConfig:
    # Gom các lỗi cấu hình vào một ValueError trước khi huấn luyện
    errors: list[str] = []

    def non_empty(path: str, value: str) -> None:
        if not value or not value.strip():
            errors.append(f"'{path}' không được để trống")

    def at_least(path: str, value: int, minimum: int) -> None:
        if value < minimum:
            errors.append(f"'{path}' phải >= {minimum}, nhận được {value}")

    def in_range(
        path: str,
        value: float,
        minimum: float,
        maximum: float,
        *,
        include_maximum: bool = True,
    ) -> None:
        upper_ok = value <= maximum if include_maximum else value < maximum
        if value < minimum or not upper_ok:
            right = "]" if include_maximum else ")"
            errors.append(
                f"'{path}' phải nằm trong [{minimum}, {maximum}{right}, nhận được {value}"
            )

    non_empty("phase.name", config.phase.name)
    non_empty("model.name_or_path", config.model.name_or_path)
    at_least("model.max_parameters", config.model.max_parameters, 1)
    if config.model.dropout is not None:
        in_range("model.dropout", config.model.dropout, 0.0, 1.0, include_maximum=False)

    at_least("data.max_source_length", config.data.max_source_length, 1)
    at_least("data.max_target_length", config.data.max_target_length, 1)
    if config.data.max_train_samples is not None:
        at_least("data.max_train_samples", config.data.max_train_samples, 1)
    if config.data.max_eval_samples is not None:
        at_least("data.max_eval_samples", config.data.max_eval_samples, 1)

    tc = config.training
    non_empty("training.output_dir", tc.output_dir)
    at_least("training.num_train_epochs", tc.num_train_epochs, 1)
    if tc.max_steps == 0 or tc.max_steps < -1:
        errors.append("'training.max_steps' phải là -1 hoặc một số nguyên dương")
    at_least("training.per_device_train_batch_size", tc.per_device_train_batch_size, 1)
    at_least("training.per_device_eval_batch_size", tc.per_device_eval_batch_size, 1)
    at_least("training.gradient_accumulation_steps", tc.gradient_accumulation_steps, 1)
    if tc.learning_rate <= 0:
        errors.append("'training.learning_rate' phải > 0")
    if tc.weight_decay < 0:
        errors.append("'training.weight_decay' phải >= 0")
    in_range("training.warmup_ratio", tc.warmup_ratio, 0.0, 1.0)
    in_range(
        "training.label_smoothing_factor",
        tc.label_smoothing_factor,
        0.0,
        1.0,
        include_maximum=False,
    )
    if tc.precision not in {"auto", "fp16", "bf16", "fp32"}:
        errors.append(
            "'training.precision' phải là một trong: auto, fp16, bf16, fp32"
        )
    non_empty("training.lr_scheduler_type", tc.lr_scheduler_type)
    non_empty("training.optim", tc.optim)

    valid_strategies = {"steps", "epoch", "no"}
    if tc.eval_strategy not in valid_strategies:
        errors.append("'training.eval_strategy' phải là: steps, epoch hoặc no")
    if tc.save_strategy not in valid_strategies:
        errors.append("'training.save_strategy' phải là: steps, epoch hoặc no")
    if tc.eval_strategy == "steps":
        at_least("training.eval_steps", tc.eval_steps, 1)
    if tc.save_strategy == "steps":
        at_least("training.save_steps", tc.save_steps, 1)
    at_least("training.save_total_limit", tc.save_total_limit, 1)
    at_least("training.logging_steps", tc.logging_steps, 1)
    at_least("training.early_stopping_patience", tc.early_stopping_patience, 0)

    if tc.early_stopping_patience > 0 and tc.eval_strategy == "no":
        errors.append("early stopping yêu cầu 'training.eval_strategy' khác 'no'")

    if tc.load_best_model_at_end:
        if tc.eval_strategy == "no" or tc.save_strategy == "no":
            errors.append(
                "load_best_model_at_end yêu cầu cả eval_strategy và save_strategy"
            )
        elif tc.eval_strategy != tc.save_strategy:
            errors.append(
                "load_best_model_at_end yêu cầu eval_strategy == save_strategy"
            )
        elif (
            tc.eval_strategy == "steps"
            and tc.eval_steps > 0
            and tc.save_steps % tc.eval_steps != 0
        ):
            errors.append(
                "load_best_model_at_end yêu cầu save_steps là bội số của eval_steps"
            )
        non_empty("training.metric_for_best_model", tc.metric_for_best_model)

    gc = config.generation
    at_least("generation.max_length", gc.max_length, 1)
    if gc.max_new_tokens is not None:
        at_least("generation.max_new_tokens", gc.max_new_tokens, 1)
    at_least("generation.min_length", gc.min_length, 0)
    if gc.min_length > gc.max_length:
        errors.append("'generation.min_length' không được lớn hơn max_length")
    at_least("generation.num_beams", gc.num_beams, 1)
    if gc.num_beams == 1 and gc.early_stopping:
        errors.append(
            "'generation.early_stopping=true' chỉ hợp lệ khi "
            "'generation.num_beams > 1'"
        )
    at_least("generation.no_repeat_ngram_size", gc.no_repeat_ngram_size, 0)
    if gc.repetition_penalty <= 0:
        errors.append("'generation.repetition_penalty' phải > 0")

    lc = config.lora
    in_range("lora.lora_dropout", lc.lora_dropout, 0.0, 1.0, include_maximum=False)
    if lc.enabled:
        at_least("lora.r", lc.r, 1)
        at_least("lora.lora_alpha", lc.lora_alpha, 1)
        non_empty("lora.target_modules", lc.target_modules)
        if tc.freeze_encoder:
            errors.append(
                "LoRA đã tự đóng băng base model; hãy đặt "
                "'training.freeze_encoder=false'"
            )

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Cấu hình không hợp lệ:\n{details}")

    return config


# Hàm hỗ trợ nội bộ

def _build_config(raw: dict[str, Any]) -> SummarizationConfig:
    # Xây dựng SummarizationConfig từ từ điển gốc
    if not isinstance(raw, dict):
        raise ValueError("Nội dung gốc của config YAML phải là một mapping/object")

    section_types = {
        "phase": PhaseConfig,
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "generation": GenerationConfig,
        "lora": LoraConfig,
    }
    unknown_sections = sorted(set(raw) - set(section_types))
    if unknown_sections:
        raise ValueError(
            "Không rõ phần cấu hình: " + ", ".join(repr(name) for name in unknown_sections)
        )

    config = SummarizationConfig(
        phase=_build_section(PhaseConfig, raw.get("phase", {}), "phase"),
        model=_build_section(ModelConfig, raw.get("model", {}), "model"),
        data=_build_section(DataConfig, raw.get("data", {}), "data"),
        training=_build_section(TrainingConfig, raw.get("training", {}), "training"),
        generation=_build_section(
            GenerationConfig,
            raw.get("generation", {}),
            "generation",
        ),
        lora=_build_section(LoraConfig, raw.get("lora", {}), "lora"),
    )
    return validate_config(config)


def _build_section(cls: type, raw: dict[str, Any], section_name: str) -> Any:
    # Xây dựng phần dataclass và từ chối khóa sai chính tả
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"Phần '{section_name}' phải là một mapping/object")
    if not raw:
        return cls()

    valid_fields = {item.name for item in fields(cls)}
    unknown_fields = sorted(set(raw) - valid_fields)
    if unknown_fields:
        names = ", ".join(repr(name) for name in unknown_fields)
        raise ValueError(f"Không rõ trường trong phần '{section_name}': {names}")

    defaults = cls()
    type_hints = get_type_hints(cls)
    converted = {
        name: _convert_type(
            value,
            getattr(defaults, name),
            f"{section_name}.{name}",
            type_hints.get(name),
        )
        for name, value in raw.items()
    }
    return cls(**converted)


def _convert_type(
    value: Any,
    current: Any,
    field_name: str,
    expected_type: Any = None,
) -> Any:
    # Đưa giá trị về đúng kiểu của trường cấu hình
    optional = False
    origin = get_origin(expected_type)
    if origin in (Union, UnionType):
        args = get_args(expected_type)
        optional = type(None) in args
        concrete_types = [arg for arg in args if arg is not type(None)]
        if len(concrete_types) == 1:
            expected_type = concrete_types[0]

    if optional and (
        value is None
        or (isinstance(value, str) and value.strip().lower() in {"none", "null"})
    ):
        return None
    if value is None:
        raise ValueError(f"Trường '{field_name}' không được nhận giá trị null")

    target_type = type(current) if current is not None else expected_type
    if target_type in (None, Any):
        return value

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(
            f"Không thể chuyển đổi '{value}' thành bool cho trường '{field_name}'"
        )

    if target_type in (int, float) and isinstance(value, bool):
        raise ValueError(
            f"Không thể chuyển đổi bool thành số cho trường '{field_name}'"
        )

    if target_type is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"Không thể chuyển đổi '{value}' thành int cho trường '{field_name}'"
            )

    if target_type is str:
        return str(value)

    try:
        return target_type(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Không thể chuyển đổi '{value}' thành {target_type.__name__} cho trường '{field_name}'"
        ) from exc
