#!/usr/bin/env python3
# Kiểm tra nhanh bộ mã hóa trước khi huấn luyện

import argparse
import sys
from pathlib import Path

# Cho phép chạy tệp lệnh trực tiếp mà không cần cài gói
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig
from src.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra quá trình tải tokenizer cho một mô hình",
    )
    parser.add_argument(
        "--model", default="VietAI/vit5-base",
        help="Tên hoặc đường dẫn mô hình (mặc định: VietAI/vit5-base)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Thử sử dụng fast tokenizer (mặc định: slow cho T5)",
    )

    args = parser.parse_args()

    print(f"Đang kiểm tra tokenizer cho: {args.model}")
    print("-" * 50)

    is_t5 = any(k in args.model.lower() for k in ["t5", "mt5", "vit5"])
    use_fast = args.fast or not is_t5

    config = ModelConfig(
        name_or_path=args.model,
        use_fast_tokenizer=use_fast,
    )

    try:
        tokenizer = load_tokenizer(config)
    except Exception as e:
        print(f"❌ THẤT BẠI: {e}")
        sys.exit(1)

    test_texts = [
        "Xin chào Việt Nam",
        "Tóm tắt văn bản tiếng Việt sử dụng mô hình học sâu",
        "summarize: Đây là một bài viết về kinh tế Việt Nam.",
    ]

    print(f"\n✅ Đã tải tokenizer thành công!")
    print(f"   Loại (Type): {type(tokenizer).__name__}")
    print(f"   Kích thước từ vựng (Vocab size): {tokenizer.vocab_size}")
    print(f"   Token đệm (Pad token): '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"   Token kết thúc (EOS token): '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    print(f"\nKiểm tra việc mã hóa (Test encodings):")
    for text in test_texts:
        tokens = tokenizer(text, return_tensors="pt")
        ids = tokens["input_ids"][0].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  Đầu vào (Input):   '{text}'")
        print(f"  Các token (Tokens):  {len(ids)} ids")
        print(f"  Giải mã (Decoded): '{decoded}'")
        print()

    print("✅ Tất cả các bài kiểm tra đều vượt qua!")


if __name__ == "__main__":
    main()
