#!/usr/bin/env python3
# Giao diện dòng lệnh để sinh tóm tắt từ điểm kiểm tra đã huấn luyện

import sys
from pathlib import Path

# Cho phép chạy tệp lệnh trực tiếp mà không cần cài gói
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.predict import main

if __name__ == "__main__":
    main()
