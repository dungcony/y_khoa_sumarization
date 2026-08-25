# Đọc và ghi các tệp kết quả đánh giá

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils import save_json, setup_logger

logger = setup_logger(__name__)


def export_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    tokenizer: Any,
    dataset: Any,
    output_path: Path,
) -> None:
    # Xuất văn bản gốc, tham chiếu và dự đoán ra JSONL
    pad_id = tokenizer.pad_token_id or 0

    predictions = np.where(predictions < 0, pad_id, predictions)
    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )
    labels = np.where(labels == -100, pad_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, (prediction, reference) in enumerate(
            zip(decoded_predictions, decoded_labels)
        ):
            article = dataset[index]["article"] if index < len(dataset) else ""
            record = {
                "index": index,
                "article": article,
                "reference": reference.strip(),
                "prediction": prediction.strip(),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Đã xuất các dự đoán: {output_path} "
        f"({len(decoded_predictions)} mẫu)"
    )


def summarize_results(output_root: str | Path) -> Optional[Path]:
    # Gộp chỉ số của các lần huấn luyện và chọn lần có ROUGE-L cao nhất
    output_root = Path(output_root)
    if not output_root.exists():
        logger.warning(f"Không tìm thấy thư mục gốc: {output_root}")
        return None

    results = []
    for run_dir in sorted(output_root.iterdir()):
        metrics = _load_run_metrics(run_dir)
        if metrics is None:
            continue

        results.append({
            "run": run_dir.name,
            "rouge1": _read_rouge(metrics, "rouge1"),
            "rouge2": _read_rouge(metrics, "rouge2"),
            "rougeL": _read_rouge(metrics, "rougeL"),
        })

    if not results:
        logger.info("Không tìm thấy kết quả nào để tóm tắt")
        return None

    results.sort(key=lambda result: result["rougeL"], reverse=True)
    csv_path = _write_csv_summary(output_root, results)
    _write_markdown_summary(output_root, results)

    best_result = results[0]
    save_json(best_result, output_root / "best_run.json")
    logger.info(f"Đã lưu tóm tắt kết quả tới: {csv_path}")
    logger.info(
        f"Lần chạy tốt nhất: {best_result['run']} "
        f"(ROUGE-L: {best_result['rougeL']:.2f})"
    )
    return csv_path


def _load_run_metrics(run_dir: Path) -> dict[str, Any] | None:
    # Tìm tệp metrics phù hợp trong một thư mục chạy
    if not run_dir.is_dir():
        return None

    for filename in (
        "eval_results.json",
        "validation_metrics.json",
        "test_metrics.json",
    ):
        metrics_path = run_dir / filename
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                return json.load(metrics_file)
    return None


def _read_rouge(metrics: dict[str, Any], metric_name: str) -> float:
    # Hỗ trợ tên chỉ số hiện tại và các tệp kết quả cũ
    for prefix in ("eval_", "validation_", "test_", ""):
        key = f"{prefix}{metric_name}"
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _write_csv_summary(
    output_root: Path,
    results: list[dict[str, Any]],
) -> Path:
    # Ghi bảng kết quả dạng CSV
    csv_path = output_root / "summary_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run", "rouge1", "rouge2", "rougeL"],
        )
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def _write_markdown_summary(
    output_root: Path,
    results: list[dict[str, Any]],
) -> None:
    # Ghi bảng kết quả dạng Markdown
    markdown_path = output_root / "summary_results.md"
    with markdown_path.open("w", encoding="utf-8") as markdown_file:
        markdown_file.write("# Kết quả tóm tắt\n\n")
        markdown_file.write("| Run | ROUGE-1 | ROUGE-2 | ROUGE-L |\n")
        markdown_file.write("|-----|---------|---------|--------|\n")
        for result in results:
            markdown_file.write(
                f"| {result['run']} | {result['rouge1']:.2f} | "
                f"{result['rouge2']:.2f} | {result['rougeL']:.2f} |\n"
            )
