# Nạp và mã hóa dữ liệu CSV/Parquet cho mô hình chuỗi sang chuỗi

from __future__ import annotations

from glob import glob
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict

from src.config import DataConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


# Làm sạch văn bản

def clean_text(text: str) -> str:
    # Chuẩn hóa Unicode và khoảng trắng
    if not text:
        return ""
    # Giữ cách biểu diễn dấu tiếng Việt nhất quán
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Nạp tập dữ liệu

def load_dataset_from_files(
        train_file: Optional[str | Path] = None,
        valid_file: Optional[str | Path] = None,
        test_file: Optional[str | Path] = None,
) -> DatasetDict:
    # Nạp các tập con CSV/Parquet được truyền vào
    from datasets import concatenate_datasets, load_dataset

    patterns = {}
    if train_file:
        patterns["train"] = str(train_file)
    if valid_file:
        patterns["validation"] = str(valid_file)
    if test_file:
        patterns["test"] = str(test_file)
    if not patterns:
        raise ValueError("Phải cung cấp ít nhất một file/pattern dữ liệu.")

    def resolve_files(split_name: str, pattern: str) -> list[Path]:
        paths = sorted(
            {Path(match) for match in glob(pattern, recursive=True) if Path(match).is_file()}
        )
        if not paths:
            raise FileNotFoundError(
                f"Không tìm thấy file cho split '{split_name}' với pattern: {pattern}"
            )
        unsupported = [
            path
            for path in paths
            if path.suffix.casefold() not in {".csv", ".parquet", ".pq"}
        ]
        if unsupported:
            names = ", ".join(str(path) for path in unsupported)
            raise ValueError(
                f"Split '{split_name}' có định dạng không hỗ trợ: {names}. "
                "Chỉ hỗ trợ CSV, Parquet hoặc PQ."
            )
        return paths

    def validate_columns(split_name: str, split_data: Dataset) -> Dataset:
        if "article" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'article' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}. "
                "Hãy chạy scripts/clean_data.py để chuẩn hóa dữ liệu trước."
            )
        if "summary" not in split_data.column_names:
            raise ValueError(
                f"Thiếu cột 'summary' trong tập {split_name}. "
                f"Các cột hiện có: {split_data.column_names}. "
                "Hãy chạy scripts/clean_data.py để chuẩn hóa dữ liệu trước."
            )
        return split_data

    resolved_files = {
        split_name: resolve_files(split_name, pattern)
        for split_name, pattern in patterns.items()
    }
    logger.info(
        "Đang tải dữ liệu: "
        + "; ".join(
            f"{split}={len(paths)} file" for split, paths in resolved_files.items()
        )
    )

    splits: dict[str, Dataset] = {}
    for split_name, paths in resolved_files.items():
        files_by_format: dict[str, list[str]] = {"csv": [], "parquet": []}
        for path in paths:
            file_format = "csv" if path.suffix.casefold() == ".csv" else "parquet"
            files_by_format[file_format].append(str(path))

        loaded_parts: list[Dataset] = []
        for file_format in ("csv", "parquet"):
            files = files_by_format[file_format]
            if not files:
                continue
            loaded = load_dataset(
                file_format,
                data_files={split_name: files},
                split=split_name,
            )
            loaded_parts.append(validate_columns(split_name, loaded))

        splits[split_name] = (
            loaded_parts[0]
            if len(loaded_parts) == 1
            else concatenate_datasets(loaded_parts)
        )

    dataset = DatasetDict(splits)

    logger.info(
        f"Đã tải tập dữ liệu: "
        f"{len(dataset.get('train', []))} train, "
        f"{len(dataset.get('validation', []))} validation"
        + (f", {len(dataset.get('test', []))} test" if 'test' in dataset else "")
    )

    return dataset


# Mã hóa dữ liệu

def preprocess_for_seq2seq(
        dataset: DatasetDict,
        tokenizer: Any,
        data_config: DataConfig,
) -> DatasetDict:
    # Mã hóa bài viết và bản tóm tắt cho Seq2SeqTrainer
    prefix = data_config.source_prefix or ""

    def tokenize_function(examples: dict[str, list]) -> dict[str, list]:
        # Mã hóa một lô dữ liệu
        inputs = [
            prefix + clean_text(article)
            for article in examples["article"]
        ]
        targets = [
            clean_text(summary)
            for summary in examples["summary"]
        ]

        model_inputs = tokenizer(
            inputs,
            max_length=data_config.max_source_length,
            truncation=True,
            padding=False,  # Đệm động tại bộ gom dữ liệu
        )

        labels = tokenizer(
            text_target=targets,
            max_length=data_config.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = DatasetDict()

    for split_name, split_data in dataset.items():
        # Giới hạn số mẫu cho lần kiểm tra nhanh nếu được cấu hình
        max_samples = None
        if split_name == "train" and data_config.max_train_samples:
            max_samples = min(data_config.max_train_samples, len(split_data))
        elif split_name in ("validation", "test") and data_config.max_eval_samples:
            max_samples = min(data_config.max_eval_samples, len(split_data))

        if max_samples:
            split_data = split_data.select(range(max_samples))
            logger.info(f"Đang giới hạn phần {split_name} xuống còn {max_samples} mẫu")

        tokenized[split_name] = split_data.map(
            tokenize_function,
            batched=True,
            remove_columns=split_data.column_names,
            desc=f"Tokenizing {split_name}",
        )

        logger.info(
            f"{split_name}: đã tokenize {len(tokenized[split_name])} mẫu"
        )

    return tokenized


def load_and_preprocess(
        tokenizer: Any,
        data_config: DataConfig,
) -> DatasetDict:
    # Nạp rồi mã hóa toàn bộ các tập con đã cấu hình
    dataset = load_dataset_from_files(
        train_file=data_config.train_file,
        valid_file=data_config.valid_file,
        test_file=data_config.test_file if data_config.test_file else None,
    )

    return preprocess_for_seq2seq(dataset, tokenizer, data_config)
