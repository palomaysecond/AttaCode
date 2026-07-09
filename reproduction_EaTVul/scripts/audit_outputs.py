from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.logging_utils import write_json


BAD_PATTERNS = [
    r"sorry,\s*to prevent abuse",
    r"free resources",
    r"not been recharged",
    r"console\.aihubmix\.com/topup",
    r"insufficient (balance|quota|credit)",
    r"rate limit",
    r"too many requests",
    r"content filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit EaTVul outputs for provider-error contamination.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "eatvul"))
    parser.add_argument("--fail-on-errors", action="store_true", help="Exit nonzero when contamination is found.")
    parser.add_argument("--max-report-files", type=int, default=50)
    return parser.parse_args()


def iter_files(output_dir: Path) -> Iterable[Path]:
    targets = [
        output_dir / "cache" / "llm",
        output_dir / "attack_pool",
        output_dir / "runs",
        output_dir / "summary",
    ]
    for target in targets:
        if target.is_file():
            yield target
        elif target.is_dir():
            yield from target.rglob("*")


def scan_file(path: Path, pattern: re.Pattern[str]) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle, start=1):
                if pattern.search(line):
                    matches.append({"line": line_no, "preview": line.strip()[:220]})
                    if len(matches) >= 5:
                        break
    except OSError as exc:
        matches.append({"line": 0, "preview": f"read error: {exc}"})
    return {"path": str(path), "matches": matches, "match_count_capped": len(matches)}


def count_error_cache(output_dir: Path) -> int:
    error_cache = output_dir / "cache" / "llm_errors"
    if not error_cache.exists():
        return 0
    return len(list(error_cache.glob("*.json")))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    pattern = re.compile("|".join(f"(?:{item})" for item in BAD_PATTERNS), re.IGNORECASE)
    contaminated = []
    contaminated_count = 0
    scanned = 0
    for path in iter_files(output_dir):
        if path.is_dir() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".log", ".txt"}:
            continue
        scanned += 1
        result = scan_file(path, pattern)
        if result["matches"]:
            contaminated_count += 1
            if len(contaminated) < args.max_report_files:
                contaminated.append(result)

    report = {
        "output_dir": str(output_dir),
        "scanned_files": scanned,
        "contaminated_files": contaminated_count,
        "reported_contaminated_files": len(contaminated),
        "llm_error_cache_files": count_error_cache(output_dir),
        "contaminated": contaminated,
    }
    report_path = output_dir / "audit" / "audit_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_errors and contaminated_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
