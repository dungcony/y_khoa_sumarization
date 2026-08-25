# Tiện ích dùng chung cho nhật ký, đọc ghi tệp và phần cứng

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml


# Ghi nhật ký

def setup_logger(
    name: str = "src",
    level: int = logging.INFO,
) -> logging.Logger:
    # Tạo bộ ghi nhật ký dùng chung cho các mô-đun
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    # Kaggle/Jupyter thường đã cấu hình bộ ghi nhật ký gốc, nếu không chặn lan truyền
    # thì mỗi thông báo sẽ xuất hiện hai lần với định dạng riêng và INFO:name
    logger.propagate = False
    return logger


# Khả năng tái lập

def set_seed(seed: int = 42) -> None:
    # Đặt hạt giống ngẫu nhiên cho Python, NumPy và PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ưu tiên khả năng tái lập hơn tốc độ tự tinh chỉnh
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Đọc và ghi tệp

def load_yaml(path: str | Path) -> dict[str, Any]:
    # Đọc tệp YAML thành từ điển
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file YAML: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:
    # Ghi dữ liệu ra JSON và tự tạo thư mục cha
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return path


def load_json(path: str | Path) -> Any:
    # Đọc tệp JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Xử lý đường dẫn

def resolve_data_path(
    file_path: str,
    search_dirs: Optional[list[str | Path]] = None,
) -> Path:
    # Tìm tệp theo đường dẫn gốc, thư mục hiện tại rồi đến search_dirs
    path = Path(file_path)

    if path.is_absolute() and path.exists():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    for search_dir in (search_dirs or []):
        candidate = Path(search_dir) / path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu: '{file_path}'\n"
        f"Đã tìm trong:\n"
        f"  - {path} (tuyệt đối)\n"
        f"  - {cwd_path} (tương đối so với CWD)\n"
        + "\n".join(f"  - {Path(d) / path}" for d in (search_dirs or []))
    )


# Tiện ích cho mô hình

def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    # Đếm tổng số tham số, số tham số được huấn luyện và bị đóng băng
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": round(100 * trainable / total, 2) if total > 0 else 0,
    }


def detect_precision() -> str:
    # Chọn bf16, fp16 hoặc fp32 theo phần cứng hiện tại
    try:
        import torch_xla.core.xla_model as xm
        _ = xm.xla_device()
        return "bf16"
    except Exception:
        pass

    if not torch.cuda.is_available():
        return "fp32"

    if torch.cuda.is_bf16_supported():
        return "bf16"

    return "fp16"


def get_device_info() -> dict[str, Any]:
    # Lấy thông tin GPU, TPU hoặc CPU hiện tại
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return {
            "device": "tpu",
            "tpu_available": True,
            "cuda_available": False,
            "num_gpus": 0,
            "gpu_names": [],
            "tpu_device": str(device),
            "precision": "bf16",
        }
    except Exception:
        pass

    info = {
        "cuda_available": torch.cuda.is_available(),
        "tpu_available": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "precision": detect_precision(),
    }

    for i in range(info["num_gpus"]):
        info["gpu_names"].append(torch.cuda.get_device_name(i))

    return info


# Định dạng

def format_number(n: int) -> str:
    # Thêm dấu phân cách hàng nghìn
    return f"{n:,}"


def format_duration(seconds: float) -> str:
    # Đổi số giây sang chuỗi giờ, phút, giây
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)
