"""Token usage reporter for autoresearch LLM-mode sessions.

Reads Claude Code's own session transcripts (stored at
`~/.claude/projects/<encoded-cwd>/*.jsonl`) and appends a per-model
cumulative token snapshot to `results/usage-<tag>.jsonl` alongside the
session artifacts defined in `program.md`.

Snapshot semantics: each invocation appends one line per distinct model
seen. Every line is a *cumulative* total as of `snapshot_at`, not a delta.
Do not sum lines — take the latest `snapshot_at` per model.

Re-running the script in the same session cannot double-count: transcripts
are the single source of truth, and each assistant message appears exactly
once on disk. The script recomputes totals from scratch on every run.

Usage:
    pixi run python tools/token_usage.py --tag <tag> [--tree-dir <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

USAGE_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def project_dir_for(tree_dir: Path) -> Path:
    """Translate an absolute path to its Claude Code transcript directory.

    Claude Code mangles the absolute path by replacing every `/` with `-`,
    so `/Users/x/y` becomes `-Users-x-y` under `~/.claude/projects/`.
    """
    abs_path = tree_dir.resolve()
    encoded = str(abs_path).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded


def _extract(record: dict) -> tuple[str, dict[str, int]] | None:
    msg = record.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    usage = msg.get("usage")
    model = msg.get("model")
    if not isinstance(usage, dict) or not isinstance(model, str) or not model:
        return None
    counters = {f: int(usage.get(f, 0) or 0) for f in USAGE_FIELDS}
    return model, counters


def aggregate_by_model(project_dir: Path) -> dict[str, dict[str, int]]:
    """Sum per-model token counts + message count across every transcript.

    Returns a dict keyed by model id; each value carries the four token
    counters plus `assistant_messages`.
    """
    totals: dict[str, dict[str, int]] = {}
    if not project_dir.is_dir():
        return totals
    # rglob, not glob: subagent transcripts live in `<session-id>/subagents/`,
    # and their tokens are billed to the session like any other.
    for jsonl in sorted(project_dir.rglob("*.jsonl")):
        with jsonl.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extracted = _extract(record)
                if extracted is None:
                    continue
                model, counters = extracted
                bucket = totals.setdefault(
                    model,
                    {field: 0 for field in USAGE_FIELDS} | {"assistant_messages": 0},
                )
                bucket["assistant_messages"] += 1
                for field, value in counters.items():
                    bucket[field] += value
    return totals


def build_records(
    tag: str,
    totals: dict[str, dict[str, int]],
    now: datetime | None = None,
) -> list[dict]:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict] = []
    for model in sorted(totals):
        rec: dict[str, object] = {"tag": tag, "snapshot_at": stamp, "model": model}
        rec.update(totals[model])
        records.append(rec)
    return records


def append_records(out_path: Path, records: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        required=True,
        help="Session tag (e.g. bev-apr23).",
    )
    parser.add_argument(
        "--tree-dir",
        type=Path,
        default=Path.cwd(),
        help="Tree root. Defaults to CWD.",
    )
    parser.add_argument(
        "--results-subdir",
        default="results",
        help="Results directory under the tree root. Default: 'results'.",
    )
    args = parser.parse_args(argv)

    project_dir = project_dir_for(args.tree_dir)

    if not project_dir.is_dir():
        print(
            f"[token_usage] no Claude Code transcripts at {project_dir} — "
            f"nothing to record. Skipping.",
            file=sys.stderr,
        )
        return 0

    totals = aggregate_by_model(project_dir)
    if not totals:
        print(
            f"[token_usage] {project_dir} has no assistant messages with "
            f"usage info. Skipping.",
            file=sys.stderr,
        )
        return 0

    out_path = args.tree_dir / args.results_subdir / f"usage-{args.tag}.jsonl"
    records = build_records(args.tag, totals)
    append_records(out_path, records)

    print(f"[token_usage] appended {len(records)} snapshot line(s) to {out_path}")
    print("[token_usage] snapshot is cumulative — take the latest per model")
    for rec in records:
        print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
