# Giao diện dòng lệnh cho chức năng sinh tóm tắt

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.predict import summarize


def build_parser() -> argparse.ArgumentParser:
    # Khai báo các tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Sinh một bản tóm tắt văn bản tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Điểm kiểm tra đầy đủ, cách gọi cũ vẫn hoạt động:
  python -m src.predict --model outputs/vit5_base/best --text "Bài viết..."

  # Điểm kiểm tra đầy đủ giai đoạn 1 và bộ điều hợp LoRA giai đoạn 2:
  python -m src.predict \\
    --base-model outputs_phase_1/vit5_base/best \\
    --adapter outputs_phase_2_lora/vit5_base/best \\
    --text "Bài viết y tế..."

  # Từ tệp:
  python -m src.predict --model outputs/vit5_base/best --file article.txt

  # Thông qua đầu vào chuẩn:
  cat article.txt | python -m src.predict --model outputs/vit5_base/best
        """,
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        dest="model_path",
        help="Full checkpoint; tên tham số cũ được giữ để tương thích",
    )
    model_group.add_argument(
        "--base-model",
        dest="base_model_path",
        help="Full base checkpoint (ví dụ checkpoint Phase 1)",
    )
    parser.add_argument(
        "--adapter",
        help=(
            "Thư mục LoRA adapter tùy chọn, phải chứa "
            "adapter_config.json; adapter không bị merge"
        ),
    )
    parser.add_argument(
        "--config",
        help="Đường dẫn tới file cấu hình YAML (tùy chọn)",
    )
    parser.add_argument("--text", help="Văn bản cần tóm tắt")
    parser.add_argument("--file", help="File chứa văn bản cần tóm tắt")
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Ghi đè tiền tố nguồn; mặc định dùng data.source_prefix "
            "trong config hoặc 'summarize: '"
        ),
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=None,
        help="Ghi đè num_beams; mặc định dùng config hoặc fallback 4",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Ghi đè max_length; mặc định dùng config hoặc fallback 200",
    )
    return parser


def read_input_text(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    # Đọc văn bản từ tham số, tệp hoặc đầu vào chuẩn
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()

    parser.error("Hãy cung cấp văn bản thông qua --text, --file, hoặc stdin")


def main() -> None:
    # Phân tích tham số và gọi API suy luận
    parser = build_parser()
    args = parser.parse_args()
    text = read_input_text(parser, args)
    config = load_config(args.config) if args.config else None

    try:
        summary = summarize(
            text=text,
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            adapter_path=args.adapter,
            config=config,
            source_prefix=args.prefix,
            num_beams=args.beams,
            max_length=args.max_length,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print("\n" + "=" * 60)
    print("BẢN TÓM TẮT:")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
