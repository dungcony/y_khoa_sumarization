#!/usr/bin/env python3
# Làm sạch và tách dữ liệu cho hai giai đoạn huấn luyện

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SPLITS = ("train", "validation", "test")
ARTICLE_ALIASES = ("article", "document", "source", "text")
SUMMARY_ALIASES = ("summary", "sumary", "target", "abstract")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
WHITESPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?P<quote>[\"”’']*)\s+")


@dataclass
class SplitReport:
    # Bộ đếm dùng khi làm sạch một tập con

    input_files: list[str] = field(default_factory=list)
    input_rows: int = 0
    dropped_quality: Counter[str] = field(default_factory=Counter)
    dropped_duplicates: int = 0
    dropped_near_duplicates: int = 0
    dropped_cross_split: int = 0
    normalized_articles: int = 0
    normalized_summaries: int = 0
    articles_with_repeated_sentences: int = 0
    repeated_sentences_removed: int = 0
    output_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_files": self.input_files,
            "input_rows": self.input_rows,
            "dropped_quality": dict(sorted(self.dropped_quality.items())),
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_near_duplicates": self.dropped_near_duplicates,
            "dropped_cross_split": self.dropped_cross_split,
            "normalized_articles": self.normalized_articles,
            "normalized_summaries": self.normalized_summaries,
            "articles_with_repeated_sentences": (
                self.articles_with_repeated_sentences
            ),
            "repeated_sentences_removed": self.repeated_sentences_removed,
            "output_rows": self.output_rows,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and clean CSV/Parquet summarization data safely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/original"),
        help=(
            "Raw data directory. Flat data/original and pre-split "
            "train/validation/test layouts are both supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clean"),
        help="New directory for cleaned data and cleaning_report.json.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "original", "split"),
        default="auto",
        help=(
            "Input layout. auto detects flat original data versus existing "
            "train/validation/test directories."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.parquet",
        help="Filename or glob matched inside every split in split layout.",
    )
    parser.add_argument(
        "--medical-variant",
        default="herding_512_bio_medicine.csv",
        help=(
            "Single medical source variant selected from flat original data. "
            "Other transformed variants are not merged."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Output format; auto uses the format of the matched input files.",
    )
    parser.add_argument(
        "--leakage-policy",
        choices=("test-priority", "train-priority", "resplit", "error"),
        default="test-priority",
        help=(
            "How to handle articles present in multiple splits. "
            "test-priority preserves held-out data; resplit hashes each article."
        ),
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("pair", "article"),
        default="pair",
        help="Drop exact article-summary pairs or keep only one row per article.",
    )
    parser.add_argument(
        "--dedupe-repeated-sentences",
        action="store_true",
        help=(
            "Remove repeated exact-normalized source sentences at least "
            "--min-repeated-sentence-chars long. This is opt-in."
        ),
    )
    parser.add_argument(
        "--min-repeated-sentence-chars",
        type=int,
        default=30,
        help="Minimum normalized sentence length eligible for internal de-duplication.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.0,
        help=(
            "Approximately find candidates and drop source documents at or above "
            "this exact word-5-gram Jaccard score; 0 disables near de-duplication."
        ),
    )
    parser.add_argument(
        "--min-article-words",
        type=int,
        default=10,
        help="Drop source articles shorter than this many whitespace words.",
    )
    parser.add_argument(
        "--min-summary-words",
        type=int,
        default=3,
        help="Drop target summaries shorter than this many whitespace words.",
    )
    parser.add_argument(
        "--max-summary-ratio",
        type=float,
        default=1.0,
        help="Maximum summary_words/article_words ratio; use 1.0 to require a shorter summary.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train proportion used only with --leakage-policy resplit.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation proportion used only with --leakage-policy resplit.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print the report without writing cleaned files.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    # Chuẩn hóa Unicode và mã đánh dấu nhưng vẫn giữ nguyên chữ tiếng Việt
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    text = ZERO_WIDTH_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = "".join(
        " " if unicodedata.category(char) == "Cc" else char
        for char in text
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def canonical_key(text: str) -> str:
    # Tạo khóa so sánh mà không đổi cách viết hoa trong dữ liệu lưu
    comparable = unicodedata.normalize("NFKC", text).casefold()
    return WHITESPACE_RE.sub(" ", comparable).strip()


def split_sentences(text: str) -> list[str]:
    # Tách câu theo dấu câu và giữ lại dấu ngoặc kép cuối câu
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        sentence = (text[start:match.start()] + match.group("quote")).strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def remove_repeated_sentences(text: str, min_chars: int = 30) -> tuple[str, int]:
    # Chỉ giữ lần xuất hiện đầu của câu dài bị lặp
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for sentence in split_sentences(text):
        key = canonical_key(sentence)
        if len(key) >= min_chars:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        kept.append(sentence)
    return " ".join(kept), removed


def find_column(columns: Iterable[Any], aliases: tuple[str, ...]) -> Any:
    lookup = {str(column).strip().casefold(): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(
        f"Required column not found. Expected one of {aliases}; "
        f"received {list(columns)}"
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        raw = pd.read_csv(path, keep_default_na=False, low_memory=False)
    elif suffix in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    article_column = find_column(raw.columns, ARTICLE_ALIASES)
    summary_column = find_column(raw.columns, SUMMARY_ALIASES)
    frame = raw[[article_column, summary_column]].copy()
    frame.columns = ["article", "summary"]
    frame["_source_file"] = str(path)
    frame["_source_row"] = raw.index
    return frame


def load_split(
    input_dir: Path,
    split: str,
    pattern: str,
    report: SplitReport,
) -> pd.DataFrame:
    split_dir = input_dir / split
    files = sorted(path for path in split_dir.glob(pattern) if path.is_file())
    report.input_files = [str(path) for path in files]
    if not files:
        return pd.DataFrame(
            columns=("article", "summary", "_source_file", "_source_row")
        )

    frames = [read_table(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    report.input_rows = len(combined)
    return combined


def apply_quality_filters(
    frame: pd.DataFrame,
    report: SplitReport,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(
            _article_key=pd.Series(dtype="object"),
            _summary_key=pd.Series(dtype="object"),
        )

    cleaned = frame.copy()
    normalized_articles = cleaned["article"].map(normalize_text)
    normalized_summaries = cleaned["summary"].map(normalize_text)
    report.normalized_articles = int(
        normalized_articles.ne(cleaned["article"].map(str)).sum()
    )
    report.normalized_summaries = int(
        normalized_summaries.ne(cleaned["summary"].map(str)).sum()
    )
    cleaned["article"] = normalized_articles
    cleaned["summary"] = normalized_summaries

    if getattr(args, "dedupe_repeated_sentences", False):
        sentence_results = cleaned["article"].map(
            lambda text: remove_repeated_sentences(
                text,
                min_chars=getattr(args, "min_repeated_sentence_chars", 30),
            )
        )
        removed_counts = sentence_results.map(lambda result: result[1])
        cleaned["article"] = sentence_results.map(lambda result: result[0])
        report.articles_with_repeated_sentences = int(removed_counts.gt(0).sum())
        report.repeated_sentences_removed = int(removed_counts.sum())
    article_words = cleaned["article"].str.split().str.len()
    summary_words = cleaned["summary"].str.split().str.len()
    article_keys = cleaned["article"].map(canonical_key)
    summary_keys = cleaned["summary"].map(canonical_key)

    reason = pd.Series("", index=cleaned.index, dtype="object")

    def reject(mask: pd.Series, label: str) -> None:
        selected = mask.fillna(False) & reason.eq("")
        reason.loc[selected] = label

    reject(cleaned["article"].eq(""), "empty_article")
    reject(cleaned["summary"].eq(""), "empty_summary")
    reject(article_words.lt(args.min_article_words), "article_too_short")
    reject(summary_words.lt(args.min_summary_words), "summary_too_short")
    reject(article_keys.eq(summary_keys), "summary_equals_article")
    reject(summary_words.ge(article_words), "summary_not_shorter_than_article")
    reject(
        summary_words.div(article_words.where(article_words.gt(0))).gt(
            args.max_summary_ratio
        ),
        "summary_too_long",
    )

    report.dropped_quality.update(reason[reason.ne("")].value_counts().to_dict())
    cleaned = cleaned.loc[reason.eq("")].copy()
    cleaned["_article_key"] = article_keys.loc[cleaned.index]
    cleaned["_summary_key"] = summary_keys.loc[cleaned.index]

    subset = ["_article_key", "_summary_key"]
    if args.dedupe_by == "article":
        subset = ["_article_key"]
    duplicate_mask = cleaned.duplicated(subset=subset, keep="first")
    report.dropped_duplicates = int(duplicate_mask.sum())
    return cleaned.loc[~duplicate_mask].reset_index(drop=True)


def article_sets(frames: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    return {
        split: set(frame["_article_key"].tolist())
        for split, frame in frames.items()
    }


def overlap_counts(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    keys = article_sets(frames)
    return {
        "train_validation": len(keys["train"] & keys["validation"]),
        "train_test": len(keys["train"] & keys["test"]),
        "validation_test": len(keys["validation"] & keys["test"]),
    }


def length_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    # Thống kê độ dài nguồn và đích trước khi ghi tệp
    if frame.empty:
        return {"rows": 0}
    article_words = frame["article"].str.split().str.len()
    summary_words = frame["summary"].str.split().str.len()
    compression = summary_words.div(article_words.where(article_words.gt(0)))

    def describe(series: pd.Series) -> dict[str, float]:
        return {
            "min": round(float(series.min()), 3),
            "p50": round(float(series.quantile(0.50)), 3),
            "p90": round(float(series.quantile(0.90)), 3),
            "p95": round(float(series.quantile(0.95)), 3),
            "p99": round(float(series.quantile(0.99)), 3),
            "max": round(float(series.max()), 3),
        }

    return {
        "rows": len(frame),
        "article_words": describe(article_words),
        "summary_words": describe(summary_words),
        "summary_article_ratio": describe(compression),
    }


def count_summary_conflicts(frames: dict[str, pd.DataFrame]) -> int:
    # Đếm bài viết có nhiều bản tóm tắt đích khác nhau
    combined = pd.concat(frames.values(), ignore_index=True)
    if combined.empty:
        return 0
    targets_per_article = combined.groupby("_article_key")["_summary_key"].nunique()
    return int(targets_per_article.gt(1).sum())


def word_ngrams(text: str, size: int = 5) -> set[int]:
    # Băm n-gram theo từ để so sánh văn bản gần trùng
    words = WORD_RE.findall(canonical_key(text))
    if len(words) < size:
        return set()
    return {
        zlib.crc32("\x1f".join(words[index:index + size]).encode("utf-8"))
        for index in range(len(words) - size + 1)
    }


def bottom_k_signature(shingles: set[int], size: int = 32) -> tuple[int, ...]:
    # Tạo chữ ký gọn để lọc ứng viên gần trùng
    return tuple(sorted(shingles)[:size])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def remove_near_duplicates(
    frames: dict[str, pd.DataFrame],
    reports: dict[str, SplitReport],
    threshold: float,
    priority: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    # Lọc văn bản gần trùng bằng bottom-k và Jaccard
    if threshold <= 0:
        return frames, []

    signature_index: dict[int, list[int]] = {}
    kept_records: list[dict[str, Any]] = []
    result: dict[str, pd.DataFrame] = {}
    examples: list[dict[str, Any]] = []

    for split in priority:
        frame = frames[split]
        keep_indices: list[Any] = []
        for index, row in frame.iterrows():
            shingles = word_ngrams(row["article"])
            signature = bottom_k_signature(shingles)
            candidate_ids: set[int] = set()
            for value in signature:
                candidate_ids.update(signature_index.get(value, ()))

            duplicate_of: dict[str, Any] | None = None
            duplicate_score = 0.0
            for candidate_id in sorted(candidate_ids):
                candidate = kept_records[candidate_id]
                if candidate["article_key"] == row["_article_key"]:
                    # Dữ liệu trùng hoàn toàn hoặc bị rò rỉ được xử lý và báo cáo riêng
                    continue
                score = jaccard(shingles, word_ngrams(candidate["article"]))
                if score >= threshold and score > duplicate_score:
                    duplicate_of = candidate
                    duplicate_score = score

            if duplicate_of is not None:
                reports[split].dropped_near_duplicates += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "dropped_split": split,
                            "dropped_file": row["_source_file"],
                            "dropped_row": int(row["_source_row"]),
                            "kept_split": duplicate_of["split"],
                            "kept_file": duplicate_of["source_file"],
                            "kept_row": duplicate_of["source_row"],
                            "word_5gram_jaccard": round(duplicate_score, 6),
                        }
                    )
                continue

            keep_indices.append(index)
            record_id = len(kept_records)
            kept_records.append(
                {
                    "article": row["article"],
                    "article_key": row["_article_key"],
                    "split": split,
                    "source_file": row["_source_file"],
                    "source_row": int(row["_source_row"]),
                }
            )
            for value in signature:
                signature_index.setdefault(value, []).append(record_id)

        result[split] = frame.loc[keep_indices].reset_index(drop=True)

    return {split: result[split] for split in SPLITS}, examples


def validate_file_selection(input_dir: Path, pattern: str) -> None:
    # Không gộp nhầm các bản biến đổi khác nhau của cùng bộ CSV
    selected_csv_names: dict[str, str] = {}
    for split in SPLITS:
        split_dir = input_dir / split
        csv_files = [
            path for path in split_dir.glob(pattern)
            if path.is_file() and path.suffix.casefold() == ".csv"
        ]
        if len(csv_files) > 1:
            names = ", ".join(sorted(path.name for path in csv_files))
            raise ValueError(
                f"Pattern matched multiple CSV variants in {split}: {names}. "
                "Do not merge these independently transformed datasets; choose one "
                "filename with --pattern. Recommended: "
                "herding_512_bio_medicine.csv."
            )
        if csv_files:
            selected_csv_names[split] = csv_files[0].name

    distinct_names = set(selected_csv_names.values())
    if len(distinct_names) > 1:
        choices = ", ".join(
            f"{split}={name}" for split, name in selected_csv_names.items()
        )
        raise ValueError(
            "CSV filename must be the same in every populated split; "
            f"received {choices}."
        )


def apply_priority_policy(
    frames: dict[str, pd.DataFrame],
    reports: dict[str, SplitReport],
    priority: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    assigned: set[str] = set()
    result: dict[str, pd.DataFrame] = {}
    for split in priority:
        frame = frames[split]
        overlap_mask = frame["_article_key"].isin(assigned)
        reports[split].dropped_cross_split = int(overlap_mask.sum())
        kept = frame.loc[~overlap_mask].reset_index(drop=True)
        result[split] = kept
        assigned.update(kept["_article_key"].tolist())
    return {split: result[split] for split in SPLITS}


def stable_bucket(article_key: str) -> float:
    digest = hashlib.sha256(article_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def deterministic_resplit(
    frames: dict[str, pd.DataFrame],
    reports: dict[str, SplitReport],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    labeled_frames: list[pd.DataFrame] = []
    for split, frame in frames.items():
        labeled = frame.copy()
        labeled["_input_split"] = split
        labeled_frames.append(labeled)
    combined = pd.concat(labeled_frames, ignore_index=True)
    subset = ["_article_key", "_summary_key"]
    if args.dedupe_by == "article":
        subset = ["_article_key"]
    duplicate_mask = combined.duplicated(subset=subset, keep="first")
    for split in SPLITS:
        removed_from_split = duplicate_mask & combined["_input_split"].eq(split)
        reports[split].dropped_duplicates += int(removed_from_split.sum())
    combined = combined.loc[~duplicate_mask].copy()

    buckets = combined["_article_key"].map(stable_bucket)
    validation_end = args.train_ratio + args.validation_ratio
    split_names = pd.Series("test", index=combined.index, dtype="object")
    split_names.loc[buckets.lt(validation_end)] = "validation"
    split_names.loc[buckets.lt(args.train_ratio)] = "train"
    return {
        split: combined.loc[split_names.eq(split)].reset_index(drop=True)
        for split in SPLITS
    }


def resolve_leakage(
    frames: dict[str, pd.DataFrame],
    reports: dict[str, SplitReport],
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    before = overlap_counts(frames)
    if args.leakage_policy == "error" and any(before.values()):
        raise ValueError(
            f"Cross-split article leakage detected: {before}. "
            "Choose test-priority, train-priority or resplit to clean it."
        )
    if args.leakage_policy == "resplit":
        cleaned = deterministic_resplit(frames, reports, args)
    elif args.leakage_policy == "train-priority":
        cleaned = apply_priority_policy(
            frames, reports, ("train", "validation", "test")
        )
    else:
        cleaned = apply_priority_policy(
            frames, reports, ("test", "validation", "train")
        )
    return cleaned, before


def infer_output_format(args: argparse.Namespace, reports: dict[str, SplitReport]) -> str:
    if args.output_format != "auto":
        return args.output_format
    suffixes = {
        Path(filename).suffix.casefold()
        for report in reports.values()
        for filename in report.input_files
    }
    if suffixes and suffixes.issubset({".parquet", ".pq"}):
        return "parquet"
    if suffixes and suffixes == {".csv"}:
        return "csv"
    raise ValueError(
        f"Cannot infer one output format from input suffixes {sorted(suffixes)}; "
        "pass --output-format explicitly."
    )


def output_paths(output_dir: Path, output_format: str) -> dict[str, Path]:
    suffix = ".parquet" if output_format == "parquet" else ".csv"
    return {
        split: output_dir / split / f"cleaned{suffix}"
        for split in SPLITS
    }


def validate_output_dir(input_dir: Path, output_dir: Path) -> None:
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError("--output-dir must be outside --input-dir.")
    if output_resolved.exists() and (
        not output_resolved.is_dir() or any(output_resolved.iterdir())
    ):
        raise FileExistsError(
            f"Output directory already exists: {output_resolved}. "
            "Choose a new or empty directory; this script never overwrites data."
        )


def write_cleaned_data(
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    output_format: str,
) -> None:
    for split in SPLITS:
        path = paths[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        public = frames[split][["article", "summary"]]
        if output_format == "parquet":
            public.to_parquet(path, index=False)
        else:
            public.to_csv(path, index=False)


def write_output_bundle(
    frames: dict[str, pd.DataFrame],
    report: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    output_format: str,
) -> None:
    # Ghi vào thư mục tạm rồi thay thế toàn bộ tập dữ liệu trong một lần
    output_resolved = output_dir.resolve()
    validate_output_dir(input_dir, output_resolved)
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_resolved.name}.tmp-",
            dir=output_resolved.parent,
        )
    )
    try:
        write_cleaned_data(
            frames,
            output_paths(temporary_dir, output_format),
            output_format,
        )
        report["output_dir"] = str(output_resolved)
        (temporary_dir / "cleaning_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_resolved)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_article_words < 1 or args.min_summary_words < 1:
        raise ValueError("Minimum word thresholds must be positive integers.")
    if not 0 < args.max_summary_ratio <= 1:
        raise ValueError("--max-summary-ratio must be in the interval (0, 1].")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be in the interval (0, 1).")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in the interval (0, 1).")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave room for test.")
    min_sentence_chars = getattr(args, "min_repeated_sentence_chars", 30)
    if min_sentence_chars < 1:
        raise ValueError("--min-repeated-sentence-chars must be positive.")
    near_threshold = getattr(args, "near_duplicate_threshold", 0.0)
    if near_threshold != 0 and not 0.5 <= near_threshold <= 1:
        raise ValueError(
            "--near-duplicate-threshold must be 0 or in the interval [0.5, 1]."
        )


def run_split_dataset(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    validate_file_selection(input_dir, args.pattern)
    reports = {split: SplitReport() for split in SPLITS}
    loaded = {
        split: load_split(input_dir, split, args.pattern, reports[split])
        for split in SPLITS
    }
    frames = {
        split: apply_quality_filters(
            loaded[split],
            reports[split],
            args,
        )
        for split in SPLITS
    }
    if frames["train"].empty or frames["validation"].empty:
        raise ValueError(
            "Both train and validation must contain rows after filtering. "
            f"Check --input-dir and --pattern ({args.pattern!r})."
        )

    summary_conflicts = count_summary_conflicts(frames)
    near_priority = ("test", "validation", "train")
    if args.leakage_policy == "train-priority":
        near_priority = ("train", "validation", "test")
    frames, near_duplicate_examples = remove_near_duplicates(
        frames,
        reports,
        threshold=getattr(args, "near_duplicate_threshold", 0.0),
        priority=near_priority,
    )
    frames, overlaps_before = resolve_leakage(frames, reports, args)
    overlaps_after = overlap_counts(frames)
    for split in SPLITS:
        reports[split].output_rows = len(frames[split])
    if frames["train"].empty or frames["validation"].empty:
        raise ValueError(
            "Cleaning would leave train or validation empty; no files were written. "
            "Review the de-duplication/leakage settings or provide more data."
        )

    output_format = infer_output_format(args, reports)
    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "output_format": output_format,
        "leakage_policy": args.leakage_policy,
        "dedupe_by": args.dedupe_by,
        "filters": {
            "min_article_words": args.min_article_words,
            "min_summary_words": args.min_summary_words,
            "max_summary_ratio": args.max_summary_ratio,
            "dedupe_repeated_sentences": getattr(
                args, "dedupe_repeated_sentences", False
            ),
            "min_repeated_sentence_chars": getattr(
                args, "min_repeated_sentence_chars", 30
            ),
            "near_duplicate_threshold": getattr(
                args, "near_duplicate_threshold", 0.0
            ),
        },
        "overlap_articles_before": overlaps_before,
        "overlap_articles_after": overlaps_after,
        "articles_with_multiple_summaries": summary_conflicts,
        "near_duplicate_detection": {
            "candidate_generation": "approximate_bottom_32_hashed_word_5grams",
            "candidate_comparison": "exact_word_5gram_jaccard",
        },
        "near_duplicate_examples": near_duplicate_examples,
        "splits": {split: reports[split].as_dict() for split in SPLITS},
        "output_length_statistics": {
            split: length_statistics(frames[split]) for split in SPLITS
        },
    }

    if not args.audit_only:
        write_output_bundle(
            frames,
            report,
            input_dir,
            args.output_dir,
            output_format,
        )

    return report


def with_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def stage_original_parquet(
    input_dir: Path,
    staging_dir: Path,
) -> dict[str, list[Path]]:
    # Ánh xạ các tệp Parquet phẳng sang bố cục theo tập con
    split_patterns = {
        "train": ("train*.parquet",),
        "validation": ("valid*.parquet", "validation*.parquet"),
        "test": ("test*.parquet",),
    }
    sources: dict[str, list[Path]] = {}
    for split, patterns in split_patterns.items():
        matches = sorted(
            {
                path.resolve()
                for pattern in patterns
                for path in input_dir.glob(pattern)
                if path.is_file()
            }
        )
        sources[split] = matches
        split_dir = staging_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source in matches:
            (split_dir / source.name).symlink_to(source)

    if not sources["train"] or not sources["validation"]:
        raise FileNotFoundError(
            "Flat original layout requires train*.parquet and valid*.parquet "
            f"files in {input_dir}."
        )
    return sources


def stage_flat_csv(
    source_path: Path,
    staging_dir: Path,
    train_ratio: float,
    validation_ratio: float,
    chunksize: int = 5_000,
) -> dict[str, int]:
    # Tách CSV theo bài viết đã chuẩn hóa và giới hạn bộ nhớ sử dụng
    if not source_path.is_file():
        raise FileNotFoundError(f"CSV source does not exist: {source_path}")

    columns = pd.read_csv(source_path, nrows=0).columns
    article_column = find_column(columns, ARTICLE_ALIASES)
    find_column(columns, SUMMARY_ALIASES)
    validation_end = train_ratio + validation_ratio
    counts = {split: 0 for split in SPLITS}
    output_files = {
        split: staging_dir / split / source_path.name for split in SPLITS
    }
    for path in output_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for chunk in pd.read_csv(
        source_path,
        chunksize=chunksize,
        keep_default_na=False,
        low_memory=False,
    ):
        article_keys = chunk[article_column].map(normalize_text).map(canonical_key)
        buckets = article_keys.map(stable_bucket)
        split_names = pd.Series("test", index=chunk.index, dtype="object")
        split_names.loc[buckets.lt(validation_end)] = "validation"
        split_names.loc[buckets.lt(train_ratio)] = "train"

        for split in SPLITS:
            selected = chunk.loc[split_names.eq(split)]
            if selected.empty:
                continue
            output_path = output_files[split]
            selected.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )
            counts[split] += len(selected)

    return counts


def rewrite_original_report(
    report: dict[str, Any],
    input_dir: Path,
    source_files: dict[str, list[Path]],
    final_output_dir: Path | None,
    written_output_dir: Path | None,
) -> None:
    report["input_dir"] = str(input_dir.resolve())
    report["source_files"] = {
        split: [str(path) for path in paths]
        for split, paths in source_files.items()
    }
    for split in SPLITS:
        report["splits"][split]["input_files"] = [
            str(path) for path in source_files.get(split, [])
        ]
    if final_output_dir is not None and written_output_dir is not None:
        report["output_dir"] = str(final_output_dir.resolve())
        (written_output_dir / "cleaning_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def run_original_data(args: argparse.Namespace) -> dict[str, Any]:
    # Làm sạch bố cục phẳng thành hai bộ dữ liệu huấn luyện riêng
    validate_args(args)
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = args.output_dir.resolve()
    temporary_output: Path | None = None
    if not args.audit_only:
        validate_output_dir(input_dir, output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.tmp-",
                dir=output_dir.parent,
            )
        )

    reports: dict[str, Any] = {}
    staging_root = Path(tempfile.mkdtemp(prefix="clean-data-stage-"))
    try:
        parquet_stage = staging_root / "parquet"
        parquet_sources = stage_original_parquet(input_dir, parquet_stage)
        parquet_written = temporary_output / "parquet" if temporary_output else None
        parquet_args = with_args(
            args,
            input_dir=parquet_stage,
            output_dir=parquet_written or output_dir / "parquet",
            pattern="*.parquet",
            output_format="parquet",
        )
        reports["parquet"] = run_split_dataset(parquet_args)
        rewrite_original_report(
            reports["parquet"],
            input_dir,
            parquet_sources,
            output_dir / "parquet" if temporary_output else None,
            parquet_written,
        )

        medical_name = getattr(
            args, "medical_variant", "herding_512_bio_medicine.csv"
        )
        medical_source = (input_dir / medical_name).resolve()
        medical_stage = staging_root / "medical"
        medical_split_counts = stage_flat_csv(
            medical_source,
            medical_stage,
            args.train_ratio,
            args.validation_ratio,
        )
        medical_sources = {
            split: [medical_source] for split in SPLITS
        }
        medical_written = temporary_output / "medical" if temporary_output else None
        medical_args = with_args(
            args,
            input_dir=medical_stage,
            output_dir=medical_written or output_dir / "medical",
            pattern=medical_source.name,
            output_format="csv",
        )
        reports["medical"] = run_split_dataset(medical_args)
        reports["medical"]["staged_split_rows"] = medical_split_counts
        rewrite_original_report(
            reports["medical"],
            input_dir,
            medical_sources,
            output_dir / "medical" if temporary_output else None,
            medical_written,
        )

        selected_files = {medical_name, "data_summary.csv"}
        skipped_variants = sorted(
            path.name
            for path in input_dir.glob("*.csv")
            if path.name not in selected_files
        )
        manifest: dict[str, Any] = {
            "layout": "original",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir) if temporary_output else None,
            "medical_variant": medical_name,
            "phase_1_dataset": "parquet news",
            "phase_2_dataset": (
                "biomedical candidate CSV; run filter_biomedical.py before training"
            ),
            "not_used_by_two_phase_pipeline": (
                ["data_summary.csv"]
                if (input_dir / "data_summary.csv").is_file()
                else []
            ),
            "skipped_transformed_variants": skipped_variants,
            "skip_reason": (
                "They are alternate transformations of the same medical sources; "
                "merging them would duplicate/overweight samples."
            ),
            "datasets": reports,
        }

        if temporary_output is not None:
            (temporary_output / "cleaning_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_output, output_dir)
            temporary_output = None
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    layout = getattr(args, "layout", "auto")
    input_dir = args.input_dir.resolve()
    if layout == "auto":
        has_split_layout = (
            (input_dir / "train").is_dir()
            and (input_dir / "validation").is_dir()
        )
        layout = "split" if has_split_layout else "original"
    if layout == "original":
        return run_original_data(args)
    return run_split_dataset(args)


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("\nAudit only: no files were written.")
    else:
        print(f"\nCleaned data written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
