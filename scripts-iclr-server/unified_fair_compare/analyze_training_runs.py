#!/usr/bin/env python3
"""Analyze unified training logs and checkpoints.

Usage:
  python scripts-iclr-server/unified_fair_compare/analyze_training_runs.py
  python scripts-iclr-server/unified_fair_compare/analyze_training_runs.py --project-root /path/to/repo
  python scripts-iclr-server/unified_fair_compare/analyze_training_runs.py --format markdown
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROGRESS_PAT = re.compile(r"Training Progress:.*?\|\s*(\d+)/(\d+)")
STEP_PAT = re.compile(r"Step\s+(\d+)/(\d+)")
METRIC_STEP_PAT = re.compile(r"step:(\d+)\s*-")
ERROR_PATTERNS = (
    "Traceback",
    "MemoryError",
    "illegal memory access",
    "CUDA out of memory",
    "RayTaskError",
    "NCCL",
)

METHOD_ORDER = {
    "Pure-GRPO": 0,
    "Intuitor": 1,
    "TokenEntropy": 2,
    "TrajectoryEntropy": 3,
    "ProbDisparity": 4,
}


@dataclass
class RunSummary:
    experiment: str
    log_path: Path
    step: int | None
    total_steps: int | None
    progress_pct: float | None
    log_mtime: str
    checkpoint_steps: list[int]
    latest_checkpoint: str
    latest_checkpoint_size: str
    latest_checkpointed_iteration: str
    error_hits: int
    log_age_minutes: float
    is_active: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze unified training status.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root path (default: current directory).",
    )
    parser.add_argument(
        "--output-root",
        default="output/self-rl-jxl",
        help="Training output root relative to project root.",
    )
    parser.add_argument(
        "--include-back",
        action="store_true",
        help="Include *-back.log files.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "markdown"),
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional output file path. If set, write report to file.",
    )
    parser.add_argument(
        "--active-window-minutes",
        type=int,
        default=20,
        help="Consider run active if log updated within N minutes (default: 20).",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f}{units[idx]}"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                continue
    return total


def parse_step(text: str) -> tuple[int | None, int | None]:
    matches = list(PROGRESS_PAT.finditer(text))
    if matches:
        cur, total = matches[-1].groups()
        return int(cur), int(total)

    matches = list(STEP_PAT.finditer(text))
    if matches:
        cur, total = matches[-1].groups()
        return int(cur), int(total)

    matches = list(METRIC_STEP_PAT.finditer(text))
    if matches:
        return int(matches[-1].group(1)), None

    return None, None


def count_errors(text: str) -> int:
    return sum(text.count(pattern) for pattern in ERROR_PATTERNS)


def iter_logs(output_root: Path, include_back: bool) -> Iterable[Path]:
    for path in sorted(output_root.glob("*/*.log")):
        if not include_back and "-back" in path.name:
            continue
        yield path


def parse_method_and_variant(experiment: str) -> tuple[str, str]:
    """Extract method and variant suffix from experiment name.

    Example:
      Unified-TokenEntropy-...-no-kl-n8 -> ("TokenEntropy", "n8")
      Unified-Intuitor-...-no-kl -> ("Intuitor", "base")
    """
    method = "Unknown"
    variant = "base"

    for candidate in METHOD_ORDER:
        if experiment.startswith(f"Unified-{candidate}-"):
            method = candidate
            break

    if experiment.endswith("-n8"):
        variant = "n8"
    elif experiment.endswith("-n12"):
        variant = "n12"
    elif "-temp" in experiment:
        temp_match = re.search(r"-temp([^-]+)$", experiment)
        variant = f"temp{temp_match.group(1)}" if temp_match else "temp"

    return method, variant


def category_sort_key(summary: RunSummary) -> tuple[int, int, str]:
    method, variant = parse_method_and_variant(summary.experiment)
    method_rank = METHOD_ORDER.get(method, 99)
    variant_rank = {"base": 0, "n8": 1, "n12": 2}.get(variant, 10)
    return method_rank, variant_rank, summary.experiment


def summarize_log(log_path: Path, active_window_minutes: int) -> RunSummary:
    text = log_path.read_text(errors="ignore")
    step, total_steps = parse_step(text)

    progress_pct = None
    if step is not None and total_steps:
        progress_pct = 100.0 * step / total_steps

    mtime_ts = log_path.stat().st_mtime
    mtime_dt = dt.datetime.fromtimestamp(mtime_ts)
    mtime = mtime_dt.strftime("%m-%d %H:%M:%S")
    age_minutes = (dt.datetime.now() - mtime_dt).total_seconds() / 60.0
    is_active = age_minutes <= active_window_minutes
    exp_dir = log_path.parent

    ckpt_dirs = sorted(
        (p for p in exp_dir.glob("global_step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    ckpt_steps = [int(p.name.split("_")[-1]) for p in ckpt_dirs]
    if ckpt_dirs:
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_name = latest_ckpt.name
        latest_ckpt_size = human_size(dir_size_bytes(latest_ckpt))
    else:
        latest_ckpt_name = "-"
        latest_ckpt_size = "-"

    tracker = exp_dir / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        latest_checkpointed_iteration = tracker.read_text(errors="ignore").strip() or "-"
    else:
        latest_checkpointed_iteration = "-"

    return RunSummary(
        experiment=exp_dir.name,
        log_path=log_path,
        step=step,
        total_steps=total_steps,
        progress_pct=progress_pct,
        log_mtime=mtime,
        checkpoint_steps=ckpt_steps,
        latest_checkpoint=latest_ckpt_name,
        latest_checkpoint_size=latest_ckpt_size,
        latest_checkpointed_iteration=latest_checkpointed_iteration,
        error_hits=count_errors(text),
        log_age_minutes=age_minutes,
        is_active=is_active,
    )


def format_progress(step: int | None, total: int | None, pct: float | None) -> tuple[str, str]:
    if step is None:
        return "-", "-"
    if total is None:
        return str(step), "-"
    return f"{step}/{total}", f"{pct:.1f}%" if pct is not None else "-"


def render_markdown(rows: list[RunSummary]) -> str:
    lines: list[str] = []
    lines.append("| Experiment | Step | Progress | Checkpoints | Latest Ckpt | Ckpt Size | Tracker | Error Hits | Log MTime | Log Age(min) | Active |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---|---:|:---:|")
    for row in rows:
        step_str, pct_str = format_progress(row.step, row.total_steps, row.progress_pct)
        ckpt_str = str(row.checkpoint_steps)
        lines.append(
            f"| {row.experiment} | {step_str} | {pct_str} | {ckpt_str} | "
            f"{row.latest_checkpoint} | {row.latest_checkpoint_size} | "
            f"{row.latest_checkpointed_iteration} | {row.error_hits} | {row.log_mtime} | "
            f"{row.log_age_minutes:.1f} | {'Y' if row.is_active else 'N'} |"
        )
    return "\n".join(lines)


def render_tsv(rows: list[RunSummary]) -> str:
    lines: list[str] = []
    lines.append(
        "experiment\tstep\tprogress\tcheckpoint_steps\tlatest_checkpoint\tlatest_ckpt_size\t"
        "latest_checkpointed_iteration\terror_hits\tlog_mtime\tlog_age_minutes\tactive\tlog_path"
    )
    for row in rows:
        step_str, pct_str = format_progress(row.step, row.total_steps, row.progress_pct)
        lines.append(
            f"{row.experiment}\t{step_str}\t{pct_str}\t{row.checkpoint_steps}\t{row.latest_checkpoint}\t"
            f"{row.latest_checkpoint_size}\t{row.latest_checkpointed_iteration}\t{row.error_hits}\t"
            f"{row.log_mtime}\t{row.log_age_minutes:.1f}\t{int(row.is_active)}\t{row.log_path}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    if not output_root.exists():
        raise SystemExit(f"Output root does not exist: {output_root}")

    rows = [summarize_log(p, args.active_window_minutes) for p in iter_logs(output_root, args.include_back)]
    rows.sort(key=category_sort_key)

    if args.format == "markdown":
        body = render_markdown(rows)
    else:
        body = render_tsv(rows)

    active_runs = sum(1 for r in rows if r.is_active)
    report = (
        f"{body}\n\n"
        f"Total runs: {len(rows)}\n"
        f"Active runs (<= {args.active_window_minutes} min): {active_runs}\n"
    )
    if args.output_file:
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Saved report to: {output_path}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()



# python scripts-iclr-server/unified_fair_compare/analyze_training_runs.py   --project-root /data_storage/wyj/jxl/ArcherCodeR   --format markdown   --output-file /data_storage/wyj/jxl/ArcherCodeR/scripts-iclr-server/unified_fair_compare/training_status_latest.md