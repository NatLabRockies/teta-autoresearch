"""Token usage reporter for autoresearch sessions.

Reads Claude Code's own session transcripts (stored at
`~/.claude/projects/<encoded-cwd>/*.jsonl`) and writes two views of what the
session cost, alongside the artifacts defined in `program.md`:

`results/usage-<tag>.jsonl` — the session view. **Appended**, one line per
model per invocation. Every line is a *cumulative* total as of `snapshot_at`,
not a delta. Do not sum lines — take the latest `snapshot_at` per model.
Re-running in the same session cannot double-count: transcripts are the single
source of truth, each assistant message appears exactly once on disk, and the
script recomputes totals from scratch every run.

`results/usage-by-exp-<tag>.jsonl` — the per-experiment view. **Overwritten**
on every run. One line per (experiment, model), holding the tokens spent
inside that experiment's window rather than a running total, so these lines
*are* meant to be summed. Attribution is retrospective: it partitions the
session at the `ended_at` timestamps the agent already records in
`results/experiments-<tag>.jsonl`, so nothing has to happen during the
experiment loop and nothing can be forgotten mid-run.

Usage:
    pixi run python tools/token_usage.py --tag <tag> [--tree-dir <path>]
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

USAGE_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# One assistant message: when it happened, which model billed it, what it cost.
# The timestamp is None for records that carry no parseable one; those still
# count toward the session total but cannot be attributed to an experiment.
Event = tuple[datetime | None, str, dict[str, int]]


def project_dir_for(tree_dir: Path) -> Path:
    """Translate an absolute path to its Claude Code transcript directory.

    Claude Code mangles the absolute path by replacing every `/` with `-`,
    so `/Users/x/y` becomes `-Users-x-y` under `~/.claude/projects/`.
    """
    abs_path = tree_dir.resolve()
    encoded = str(abs_path).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded


def parse_stamp(value: str) -> datetime | None:
    """Parse an ISO-8601 stamp, normalizing to an aware UTC datetime.

    Naive stamps are assumed UTC rather than rejected, so that comparing a
    transcript timestamp against an experiment's `ended_at` can never raise
    on a tz-aware/naive mismatch.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def root_commit_date(tree_dir: Path) -> datetime | None:
    """When this tree came into existence: the date of its scaffold commit.

    A tree path can host more than one session — an aborted setup attempt,
    then the real run. Everything at or after this stamp belongs to this tree;
    anything earlier came from something else that ran in the same directory.
    Returns None if `tree_dir` is not a git repo, which is not an error here.
    """
    try:
        root = subprocess.run(
            ["git", "-C", str(tree_dir), "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if not root:
            return None
        stamp = subprocess.run(
            ["git", "-C", str(tree_dir), "log", "-1", "--format=%aI", root[0]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    return parse_stamp(stamp) if stamp else None


def _extract(record: dict) -> Event | None:
    msg = record.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    usage = msg.get("usage")
    model = msg.get("model")
    if not isinstance(usage, dict) or not isinstance(model, str) or not model:
        return None
    stamp = record.get("timestamp")
    when = parse_stamp(stamp) if isinstance(stamp, str) else None
    counters = {f: int(usage.get(f, 0) or 0) for f in USAGE_FIELDS}
    return when, model, counters


def collect_events(project_dir: Path, since: datetime | None = None) -> list[Event]:
    """Every billed assistant message under `project_dir`, in file order.

    With `since` set, records outside the window are dropped — including
    records with no parseable timestamp, which cannot be shown to belong to
    the window.
    """
    events: list[Event] = []
    if not project_dir.is_dir():
        return events
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
                event = _extract(record)
                if event is None:
                    continue
                if since is not None and (event[0] is None or event[0] < since):
                    continue
                events.append(event)
    return events


def _empty_bucket() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS} | {"assistant_messages": 0}


def totals_by_model(events: list[Event]) -> dict[str, dict[str, int]]:
    """Sum per-model token counts + message count over a list of events."""
    totals: dict[str, dict[str, int]] = {}
    for _, model, counters in events:
        bucket = totals.setdefault(model, _empty_bucket())
        bucket["assistant_messages"] += 1
        for field, value in counters.items():
            bucket[field] += value
    return totals


def aggregate_by_model(
    project_dir: Path, since: datetime | None = None
) -> dict[str, dict[str, int]]:
    """Sum per-model token counts + message count across every transcript.

    Returns a dict keyed by model id; each value carries the four token
    counters plus `assistant_messages`.
    """
    return totals_by_model(collect_events(project_dir, since=since))


def read_experiments(path: Path) -> list[tuple[int, datetime]]:
    """Read `(exp, ended_at)` pairs from a `results/experiments-<tag>.jsonl`.

    Sorted by `ended_at`, which is what the attribution boundaries are. Lines
    missing either field, or carrying an unparseable stamp, are skipped: a
    crashed experiment the agent never finished logging should not swallow the
    tokens of the ones around it.
    """
    if not path.is_file():
        return []
    found: list[tuple[int, datetime]] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            exp = record.get("exp")
            ended = record.get("ended_at")
            if not isinstance(exp, int) or not isinstance(ended, str):
                continue
            when = parse_stamp(ended)
            if when is None:
                continue
            found.append((exp, when))
    found.sort(key=lambda pair: pair[1])
    return found


def attribute_by_experiment(
    events: list[Event], experiments: list[tuple[int, datetime]]
) -> dict[int | None, dict[str, dict[str, int]]]:
    """Partition events across experiments by timestamp.

    An event belongs to the first experiment whose `ended_at` is at or after
    it. That partitions the whole session with no gaps and no overlaps: setup
    and the baseline run fall into experiment 0's bucket, and the thinking
    that precedes an experiment falls into the experiment it produced, which
    is the honest place for it — the cost of an idea includes forming it.

    Events after the last `ended_at` (the final learnings update, the wrap-up)
    land under key `None`. Events with no timestamp are unattributable and are
    dropped here; they are still counted in the session total.
    """
    buckets: dict[int | None, dict[str, dict[str, int]]] = {}
    boundaries = [ended for _, ended in experiments]
    for when, model, counters in events:
        if when is None:
            continue
        index = bisect.bisect_left(boundaries, when)
        exp = experiments[index][0] if index < len(experiments) else None
        by_model = buckets.setdefault(exp, {})
        bucket = by_model.setdefault(model, _empty_bucket())
        bucket["assistant_messages"] += 1
        for field, value in counters.items():
            bucket[field] += value
    return buckets


def build_records(
    tag: str,
    totals: dict[str, dict[str, int]],
    now: datetime | None = None,
) -> list[dict]:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict] = []
    for model in sorted(totals):
        rec: dict[str, object] = {
            "tag": tag,
            "scope": "session",
            "snapshot_at": stamp,
            "model": model,
        }
        rec.update(totals[model])
        records.append(rec)
    return records


def build_exp_records(
    tag: str, buckets: dict[int | None, dict[str, dict[str, int]]]
) -> list[dict]:
    """One record per (experiment, model), experiments in order, tail last."""
    records: list[dict] = []
    numbered = sorted(k for k in buckets if k is not None)
    for exp in [*numbered, *([None] if None in buckets else [])]:
        for model in sorted(buckets[exp]):
            rec: dict[str, object] = {
                "tag": tag,
                "scope": "exp",
                "exp": exp,
                "model": model,
            }
            rec.update(buckets[exp][model])
            records.append(rec)
    return records


def append_records(out_path: Path, records: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_records(out_path: Path, records: list[dict]) -> None:
    """Overwrite, not append: this file is fully derived, so a rerun that
    replaces it is idempotent, and appending would duplicate every line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
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
    parser.add_argument(
        "--experiments",
        type=Path,
        help=(
            "Experiment log to attribute tokens against. Defaults to "
            "<results>/experiments-<tag>.jsonl."
        ),
    )
    parser.add_argument(
        "--since",
        help=(
            "Ignore records before this ISO-8601 timestamp. Defaults to the "
            "tree's scaffold commit date, so an earlier session that ran in "
            "this same directory is not folded in."
        ),
    )
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="Count every record in the transcript directory, with no window.",
    )
    args = parser.parse_args(argv)

    tree_dir = args.tree_dir.resolve()

    since: datetime | None = None
    if args.all_history:
        if args.since:
            print("[token_usage] --since and --all-history conflict", file=sys.stderr)
            return 1
    elif args.since:
        since = parse_stamp(args.since)
        if since is None:
            print(f"[token_usage] unparseable --since: {args.since}", file=sys.stderr)
            return 1
    else:
        since = root_commit_date(tree_dir)

    project_dir = project_dir_for(tree_dir)

    if not project_dir.is_dir():
        print(
            f"[token_usage] no Claude Code transcripts at {project_dir} — "
            f"nothing to record. Skipping.",
            file=sys.stderr,
        )
        return 0

    events = collect_events(project_dir, since=since)
    totals = totals_by_model(events)
    if not totals:
        print(
            f"[token_usage] {project_dir} has no assistant messages with "
            f"usage info. Skipping.",
            file=sys.stderr,
        )
        return 0

    results_dir = tree_dir / args.results_subdir
    out_path = results_dir / f"usage-{args.tag}.jsonl"
    records = build_records(args.tag, totals)
    append_records(out_path, records)

    print(f"[token_usage] appended {len(records)} snapshot line(s) to {out_path}")
    print("[token_usage] snapshot is cumulative — take the latest per model")
    for rec in records:
        print(json.dumps(rec))

    exp_log = args.experiments or (results_dir / f"experiments-{args.tag}.jsonl")
    experiments = read_experiments(exp_log)
    if not experiments:
        print(
            f"[token_usage] no usable experiment records in {exp_log} — "
            f"skipping the per-experiment breakdown",
            file=sys.stderr,
        )
        return 0

    buckets = attribute_by_experiment(events, experiments)
    exp_records = build_exp_records(args.tag, buckets)
    exp_path = results_dir / f"usage-by-exp-{args.tag}.jsonl"
    write_records(exp_path, exp_records)
    print(
        f"[token_usage] wrote {len(exp_records)} per-experiment line(s) across "
        f"{len(experiments)} experiment(s) to {exp_path}"
    )
    print("[token_usage] per-experiment lines are per-window totals — safe to sum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
