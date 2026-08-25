# Tính các chỉ số đánh giá cho bài toán tóm tắt

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

import numpy as np


_VIETNAMESE_ROUGE_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_vietnamese_for_rouge(text: str) -> list[str]:
    # Tách từ cho ROUGE mà vẫn giữ dấu tiếng Việt
    normalized = unicodedata.normalize("NFC", text.casefold())
    return _VIETNAMESE_ROUGE_TOKEN_PATTERN.findall(normalized)


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    # Tính ROUGE-1/2/L theo thang điểm 0-100
    import evaluate

    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
        tokenizer=tokenize_vietnamese_for_rouge,
    )

    return {
        "rouge1": round(float(results["rouge1"]) * 100, 2),
        "rouge2": round(float(results["rouge2"]) * 100, 2),
        "rougeL": round(float(results["rougeL"]) * 100, 2),
    }


def build_compute_metrics(tokenizer: Any) -> Callable:
    # Tạo hàm gọi lại ROUGE cho Seq2SeqTrainer
    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred
        pad_id = tokenizer.pad_token_id or 0

        # Loại mã âm hoặc mã nằm ngoài từ vựng trước khi giải mã
        if isinstance(predictions, np.ndarray):
            predictions = np.where(predictions < 0, pad_id, predictions)
            predictions = np.where(
                predictions >= tokenizer.vocab_size,
                pad_id,
                predictions,
            )

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        if isinstance(labels, np.ndarray):
            labels = np.where(labels == -100, pad_id, labels)

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )
        decoded_preds = [prediction.strip() for prediction in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        scores = compute_rouge(decoded_preds, decoded_labels)
        prediction_rows = (
            predictions if isinstance(predictions, np.ndarray) else [predictions]
        )
        generation_lengths = [
            np.count_nonzero(prediction != pad_id)
            for prediction in prediction_rows
        ]
        scores["gen_len"] = round(float(np.mean(generation_lengths)), 1)
        return scores

    return compute_metrics
