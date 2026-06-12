#!/usr/bin/env python3
"""Download W&B metrics for ppo_epoch ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


METRICS = [
    "val-core/livecodebench/acc/mean@4",
    "response_length/mean",
    "actor/action_reward",
    "val_token/thinking_and_reasoning",
    "internal_metrics/trajectory_entropy/mean",
    "internal_metrics/token_entropy/mean",
    "internal_metrics/self_certainty/mean",
    "internal_metrics/prob_disparity/mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download W&B ppo_epoch ablation metrics.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root.")
    parser.add_argument("--project-path", default="jxl-dragon/self-rl-jxl", help="W&B project path.")
    parser.add_argument(
        "--output-root",
        default="output/self-rl-jxl",
        help="Experiment output root relative to repo root.",
    )
    parser.add_argument(
        "--export-subdir",
        default="wandb_base_metrics/ablation-study-ppo_epoch",
        help="Export folder name under output root.",
    )
    return parser.parse_args()


def clean_sys_path(repo_root: Path) -> None:
    cleaned: list[str] = []
    for p in sys.path:
        try:
            resolved = Path(p).resolve() if p else None
        except Exception:
            resolved = None
        if p == "" or resolved == repo_root:
            continue
        cleaned.append(p)
    sys.path = cleaned


def fetch_metric_history_rows(run, metrics: list[str]) -> list[dict[str, object]]:
    """Fetch each metric separately and outer-join by _step."""
    step_rows: dict[object, dict[str, object]] = {}
    for metric in metrics:
        for row in run.scan_history(keys=["_step", metric]):
            step = row.get("_step")
            if step is None:
                continue
            if step not in step_rows:
                step_rows[step] = {"_step": step}
            value = row.get(metric, None)
            if value is not None:
                step_rows[step][metric] = value

    def _step_sort_key(v: object) -> tuple[int, object]:
        if isinstance(v, (int, float)):
            return (0, v)
        return (1, str(v))

    return [step_rows[s] for s in sorted(step_rows.keys(), key=_step_sort_key)]


def discover_ppo_epoch_runs(output_root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("Unified-"):
            continue
        if re.search(r"-ppo_epoch\d+$", name):
            names.append(name)
    return names


def parse_ppo_epoch_info(run_name: str) -> dict[str, str]:
    method = "Unknown"
    for candidate in ["Pure-GRPO", "Intuitor", "TokenEntropy", "TrajectoryEntropy", "ProbDisparity"]:
        if run_name.startswith(f"Unified-{candidate}-"):
            method = candidate
            break

    ppo_epoch = ""
    m = re.search(r"-ppo_epoch(\d+)$", run_name)
    if m:
        ppo_epoch = m.group(1)

    return {"method": method, "ppo_epoch": ppo_epoch}


def combine_tables(export_dir: Path, run_names: list[str]) -> tuple[Path, Path]:
    merged_history = export_dir / "all_ablation_ppo_epoch_runs.history.csv"
    merged_summary = export_dir / "all_ablation_ppo_epoch_runs.summary.csv"

    history_fields = ["run_name", "run_id", "method", "ppo_epoch", "_step"] + METRICS
    with merged_history.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=history_fields)
        writer.writeheader()
        for run_name in run_names:
            hist_path = export_dir / f"{run_name}.history.csv"
            summary_path = export_dir / f"{run_name}.summary.json"
            if not hist_path.exists() or not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            run_id = payload.get("run_id", "")
            method = payload.get("method", "Unknown")
            ppo_epoch = payload.get("ppo_epoch", "")
            with hist_path.open(newline="") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    out = {
                        "run_name": run_name,
                        "run_id": run_id,
                        "method": method,
                        "ppo_epoch": ppo_epoch,
                        "_step": row.get("_step", ""),
                    }
                    for m in METRICS:
                        out[m] = row.get(m, "")
                    writer.writerow(out)

    summary_fields = ["run_name", "run_id", "url", "method", "ppo_epoch"] + METRICS
    with merged_summary.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=summary_fields)
        writer.writeheader()
        for run_name in run_names:
            summary_path = export_dir / f"{run_name}.summary.json"
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            row = {
                "run_name": payload.get("run_name", run_name),
                "run_id": payload.get("run_id", ""),
                "url": payload.get("url", ""),
                "method": payload.get("method", "Unknown"),
                "ppo_epoch": payload.get("ppo_epoch", ""),
            }
            summary_metrics = payload.get("summary", {})
            for m in METRICS:
                row[m] = summary_metrics.get(m, "")
            writer.writerow(row)

    return merged_history, merged_summary


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    export_dir = output_root / args.export_subdir
    export_dir.mkdir(parents=True, exist_ok=True)

    run_names = discover_ppo_epoch_runs(output_root)
    if not run_names:
        raise SystemExit(f"No ppo_epoch ablation runs found under: {output_root}")

    clean_sys_path(repo_root)
    import wandb  # pylint: disable=import-outside-toplevel

    api = wandb.Api(timeout=120)
    runs = list(api.runs(args.project_path))
    name_to_runs: dict[str, list] = {}
    for run in runs:
        name_to_runs.setdefault(run.name, []).append(run)

    summary_rows: list[dict[str, object]] = []
    found = 0
    for name in run_names:
        info = parse_ppo_epoch_info(name)
        matched = name_to_runs.get(name, [])
        if not matched:
            summary_rows.append(
                {
                    "run_name": name,
                    "method": info["method"],
                    "ppo_epoch": info["ppo_epoch"],
                    "status": "not_found",
                    "run_id": "",
                    "history_rows": 0,
                    "url": "",
                }
            )
            continue

        run = sorted(matched, key=lambda x: x.created_at)[-1]
        keys = ["_step"] + METRICS
        joined_rows = fetch_metric_history_rows(run, METRICS)
        row_count = len(joined_rows)

        history_path = export_dir / f"{name}.history.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in joined_rows:
                writer.writerow({k: row.get(k, None) for k in keys})

        summary_path = export_dir / f"{name}.summary.json"
        payload = {
            "run_name": name,
            "run_id": run.id,
            "url": run.url,
            "method": info["method"],
            "ppo_epoch": info["ppo_epoch"],
            "summary": {k: run.summary.get(k, None) for k in METRICS},
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        summary_rows.append(
            {
                "run_name": name,
                "method": info["method"],
                "ppo_epoch": info["ppo_epoch"],
                "status": "ok",
                "run_id": run.id,
                "history_rows": row_count,
                "url": run.url,
            }
        )
        found += 1

    manifest = {
        "project_path": args.project_path,
        "metrics": METRICS,
        "ppo_epoch_runs_local": run_names,
        "downloaded_count": found,
        "total_runs": len(run_names),
        "rows": summary_rows,
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    with (export_dir / "download_summary.csv").open("w", newline="") as f:
        fieldnames = ["run_name", "method", "ppo_epoch", "status", "run_id", "history_rows", "url"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    merged_history, merged_summary = combine_tables(export_dir, run_names)

    print(f"ppo_epoch runs detected: {len(run_names)}")
    print(f"Downloaded: {found}")
    print(f"Export dir: {export_dir}")
    print(f"Merged history table: {merged_history}")
    print(f"Merged summary table: {merged_summary}")


if __name__ == "__main__":
    main()
