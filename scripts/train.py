#!/usr/bin/env python3
# Giao diện dòng lệnh để huấn luyện mô hình từ cấu hình YAML

import argparse
import sys
from pathlib import Path

# Cho phép chạy tệp lệnh trực tiếp mà không cần cài gói
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, apply_overrides
from src.trainer import train

def main():
    # Chạy huấn luyện từ cấu hình YAML và các giá trị ghi đè dòng lệnh

    args, overrides = parse_arguments()

    config = load_config(args.config)

    if overrides:
        config = apply_overrides(config, overrides)

    metrics = train(config)

    print("\n" + "=" * 50)
    print("CÁC CHỈ SỐ CUỐI CÙNG (FINAL METRICS):")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 50)


def parse_arguments() -> tuple[argparse.Namespace, dict]:
    # Đọc tham số dòng lệnh và tách các giá trị ghi đè cấu hình
    parser = argparse.ArgumentParser(
        description="Huấn luyện một mô hình tóm tắt văn bản tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/vit5_base.yaml)",
    )

    # Các giá trị ghi đè thường dùng
    parser.add_argument("--data-dir", help="Thư mục chứa các file parquet train/valid")
    parser.add_argument("--train-file", help="Đường dẫn tới file parquet huấn luyện")
    parser.add_argument("--valid-file", help="Đường dẫn tới file parquet đánh giá")
    parser.add_argument("--output-dir", help="Thư mục đầu ra cho model")
    parser.add_argument("--epochs", type=int, help="Số lượng vòng học (epochs)")
    parser.add_argument("--max-steps", type=int, help="Số bước huấn luyện tối đa (ghi đè epochs)")
    parser.add_argument("--learning-rate", type=float, help="Tốc độ học (learning rate)")
    parser.add_argument("--batch-size", type=int, help="Số lượng bài báo học cùng lúc")
    parser.add_argument("--seed", type=int, help="Hạt giống ngẫu nhiên (seed)")
    parser.add_argument("--resume", help="Đường dẫn tới bản lưu (checkpoint) để chạy tiếp")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", 
                        help="Ghi đè nâng cao (vd: --set training.warmup_ratio=0.05)")

    args = parser.parse_args()
    overrides = {}

    # Ánh xạ cờ dòng lệnh sang khóa cấu hình dạng dấu chấm
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_files = list(data_dir.glob("train*.parquet"))
        valid_files = list(data_dir.glob("valid*.parquet"))
        if train_files: overrides["data.train_file"] = str(train_files[0])
        if valid_files: overrides["data.valid_file"] = str(valid_files[0])

    if args.train_file: overrides["data.train_file"] = args.train_file
    if args.valid_file: overrides["data.valid_file"] = args.valid_file
    if args.output_dir: overrides["training.output_dir"] = args.output_dir
    if args.epochs: overrides["training.num_train_epochs"] = args.epochs
    if args.max_steps: overrides["training.max_steps"] = args.max_steps
    if args.learning_rate: overrides["training.learning_rate"] = args.learning_rate
    if args.batch_size: overrides["training.per_device_train_batch_size"] = args.batch_size
    if args.seed: overrides["training.seed"] = args.seed
    if args.resume: overrides["training.resume_from_checkpoint"] = args.resume

    if args.set:
        for item in args.set:
            if "=" not in item:
                parser.error(f"Định dạng --set không hợp lệ: '{item}'. Hãy dùng KEY=VALUE")
            key, value = item.split("=", 1)
            overrides[key] = value

    return args, overrides


if __name__ == "__main__":
    main()
