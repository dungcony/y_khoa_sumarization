# Callback theo dõi tiến trình huấn luyện

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from src.utils import format_duration


class TrainingProgressCallback(TrainerCallback):
    # In tiến trình định kỳ ra màn hình và tệp nhật ký

    def __init__(
        self,
        label: str,
        log_every_steps: int = 10,
        heartbeat_seconds: int = 60,
        log_file: str | Path | None = None,
    ) -> None:
        self.label = label
        self.log_every_steps = max(1, int(log_every_steps))
        self.heartbeat_seconds = max(10, int(heartbeat_seconds))
        self.log_file = Path(log_file) if log_file else None
        self.started_at: float | None = None
        self.last_completed_step = 0
        self.max_steps = 0
        self._is_main_process = True
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None

    def _device_status(self) -> str:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return f"GPU RAM={allocated:.2f}/{reserved:.2f} GB"
        return "CPU đang hoạt động"

    def _emit(self, message: str) -> None:
        if not self._is_main_process:
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{self.label}] {message}"
        with self._write_lock:
            print(line, flush=True)
            if self.log_file is not None:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

    def _progress_text(self, step: int | None = None) -> str:
        step = self.last_completed_step if step is None else step
        if self.started_at is None:
            return f"bước={step}/{self.max_steps}"

        elapsed = time.time() - self.started_at
        percent = 100 * step / self.max_steps if self.max_steps else 0
        eta_text = "đang tính"
        if step > 0 and self.max_steps > step:
            eta = elapsed / step * (self.max_steps - step)
            eta_text = format_duration(eta)

        return (
            f"bước={step}/{self.max_steps} ({percent:.1f}%) | "
            f"đã chạy={format_duration(elapsed)} | còn lại={eta_text}"
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            self._emit(
                f"Vẫn đang huấn luyện | {self._progress_text()} | "
                f"{self._device_status()}"
            )

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.time()
        self.last_completed_step = state.global_step
        self.max_steps = state.max_steps
        self._is_main_process = state.is_world_process_zero
        self._stop_event.clear()
        self._emit(f"Bắt đầu huấn luyện | {self._progress_text()}")

        if self._is_main_process:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"{self.label}-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        epoch_number = int(state.epoch or 0) + 1
        self._emit(f"Bắt đầu epoch {epoch_number}/{int(args.num_train_epochs)}")

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        next_step = state.global_step + 1
        if next_step == 1 or next_step % self.log_every_steps == 0:
            self._emit(f"Đang xử lý bước {next_step}/{state.max_steps}")

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.last_completed_step = state.global_step
        if state.global_step == 1 or state.global_step % self.log_every_steps == 0:
            self._emit(f"Đã hoàn tất | {self._progress_text()} | {self._device_status()}")

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        fields = []
        for key in ("loss", "learning_rate", "grad_norm", "epoch"):
            if logs and key in logs:
                value = logs[key]
                formatted = f"{value:.6g}" if isinstance(value, (int, float)) else value
                fields.append(f"{key}={formatted}")
        if fields:
            self._emit("Chỉ số | " + " | ".join(fields))

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        summary = ", ".join(
            f"{key}={value:.4f}"
            for key, value in (metrics or {}).items()
            if key in {"eval_loss", "eval_rouge1", "eval_rouge2", "eval_rougeL"}
        )
        self._emit(f"Đánh giá xong | {summary or 'không có chỉ số tóm tắt'}")

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._emit(f"Đã lưu điểm kiểm tra tại bước {state.global_step}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.last_completed_step = state.global_step
        self._emit(f"Kết thúc huấn luyện | {self._progress_text()}")
        self.stop()
