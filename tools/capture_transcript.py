"""Capture Claude Code session transcripts for an experiment session.

Claude Code writes its own transcripts to `~/.claude/projects/<encoded-cwd>/`.
They record every tool call the agent made, every file it read, and every
word the operator typed — none of which is reconstructable from the results
TSV or the JSONL the agent writes about itself.

Two reasons to keep them alongside the results:

1. **Isolation is a claim, and this is the evidence.** A tree is supposed to
   be an independent sample. The transcript shows whether the agent actually
   stayed inside it, so `--audit` classifies every path it touched as in-tree
   or out-of-tree and lists the exceptions.
2. **Operator input is a variable.** If one arm is told more than another,
   that is direction, not method. The transcript is the only complete record
   of what was said during a session.

Copies are snapshots: re-running overwrites, so a mid-session capture is
safe and a session-end capture supersedes it.

Usage:
    pixi run python tools/capture_transcript.py --tag <tag> [--tree-dir <path>]
    pixi run python tools/capture_transcript.py --tag <tag> --audit-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from token_usage import project_dir_for  # noqa: E402

# Tool inputs that name a path directly.
PATH_KEYS = ("file_path", "notebook_path", "path", "file")

# Absolute-looking paths inside a shell command.
_ABS_PATH_RE = re.compile(r"(?<![\w=])(/[A-Za-z0-9._\-/]+)")

# Shell constructs that reach outside the working directory without naming
# an absolute path.
_PARENT_ESCAPE_RE = re.compile(
    r"(?:^|[\s\"'`(=])(\.\./[A-Za-z0-9._\-/]*|\.\.)(?=[\s\"'`);|&]|$)"
)


def iter_records(project_dir: Path):
    """Yield (transcript_path, record) for every JSON line, subagents included."""
    for jsonl in sorted(project_dir.rglob("*.jsonl")):
        with jsonl.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield jsonl, json.loads(line)
                except json.JSONDecodeError:
                    continue


def tool_uses(record: dict) -> list[tuple[str, dict]]:
    msg = record.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            inp = block.get("input")
            if isinstance(name, str) and isinstance(inp, dict):
                out.append((name, inp))
    return out


def user_text(record: dict) -> str | None:
    msg = record.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def paths_referenced(name: str, inp: dict) -> list[str]:
    """Best-effort extraction of filesystem paths from a tool call."""
    found: list[str] = []
    for key in PATH_KEYS:
        value = inp.get(key)
        if isinstance(value, str) and value:
            found.append(value)
    command = inp.get("command")
    if isinstance(command, str):
        found.extend(_ABS_PATH_RE.findall(command))
        found.extend(m.group(1) for m in _PARENT_ESCAPE_RE.finditer(command))
    return found


def is_outside(path_str: str, tree_root: Path) -> bool:
    """True if the path resolves outside the tree.

    Relative paths resolve against the tree root, which is where a session
    runs. `..` escapes are caught by resolving before comparing.
    """
    try:
        candidate = Path(path_str).expanduser()
        resolved = (
            candidate if candidate.is_absolute() else (tree_root / candidate)
        ).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        resolved.relative_to(tree_root.resolve())
        return False
    except ValueError:
        return True


def audit(project_dir: Path, tree_root: Path) -> dict:
    tools: Counter[str] = Counter()
    assistant_messages = 0
    prompts: list[str] = []
    outside: list[tuple[str, str]] = []
    seen_outside: set[tuple[str, str]] = set()
    timestamps: list[str] = []
    transcripts: set[Path] = set()

    for jsonl, record in iter_records(project_dir):
        transcripts.add(jsonl)
        stamp = record.get("timestamp")
        if isinstance(stamp, str):
            timestamps.append(stamp)

        msg = record.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant_messages += 1

        text = user_text(record)
        if text and not text.startswith("<"):
            prompts.append(text.strip())

        for name, inp in tool_uses(record):
            tools[name] += 1
            for raw in paths_referenced(name, inp):
                if is_outside(raw, tree_root):
                    key = (name, raw)
                    if key not in seen_outside:
                        seen_outside.add(key)
                        outside.append(key)

    timestamps.sort()
    return {
        "transcripts": sorted(transcripts),
        "assistant_messages": assistant_messages,
        "tools": tools,
        "prompts": prompts,
        "outside": outside,
        "first": timestamps[0] if timestamps else None,
        "last": timestamps[-1] if timestamps else None,
    }


def render(tag: str, tree_root: Path, project_dir: Path, result: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# Session transcript audit — {tag}")
    add("")
    add(f"- tree: `{tree_root}`")
    add(f"- transcripts: `{project_dir}`")
    add(f"- captured: {datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')}")
    add(f"- session span: {result['first']} → {result['last']}")
    add(f"- assistant messages: {result['assistant_messages']}")
    add(f"- transcript files: {len(result['transcripts'])}")
    add("")

    add("## Tool calls")
    add("")
    if result["tools"]:
        add("| tool | calls |")
        add("| --- | --- |")
        for name, count in result["tools"].most_common():
            add(f"| `{name}` | {count} |")
    else:
        add("None recorded.")
    add("")

    add("## Operator input")
    add("")
    add(
        "Everything the human typed. In a two-arm study this is the variable to "
        "hold constant — anything beyond the session-start prompt is direction."
    )
    add("")
    if result["prompts"]:
        for i, prompt in enumerate(result["prompts"], 1):
            add(f"{i}. {prompt}")
    else:
        add("None recorded.")
    add("")

    add("## Paths touched outside the tree")
    add("")
    if result["outside"]:
        add(
            f"**{len(result['outside'])} distinct out-of-tree reference(s).** Each needs a "
            "look: a path outside the tree is how prior findings would enter an "
            "otherwise-isolated run. Extraction is best-effort and over-reports "
            "(a `/usr/bin` tool path counts), so read before concluding."
        )
        add("")
        add("| tool | path |")
        add("| --- | --- |")
        for name, path in result["outside"]:
            add(f"| `{name}` | `{path}` |")
    else:
        add("None. Every path referenced resolves inside the tree.")
    add("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="Session tag (e.g. bev-apr23).")
    parser.add_argument(
        "--tree-dir", type=Path, default=Path.cwd(), help="Tree root. Defaults to CWD."
    )
    parser.add_argument(
        "--results-subdir",
        default="results",
        help="Results directory under the tree root. Default: 'results'.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write the audit but do not copy the raw transcripts.",
    )
    args = parser.parse_args(argv)

    tree_root = args.tree_dir.resolve()
    project_dir = project_dir_for(tree_root)

    if not project_dir.is_dir():
        print(
            f"[capture_transcript] no Claude Code transcripts at {project_dir} — "
            f"nothing to capture. Skipping.",
            file=sys.stderr,
        )
        return 0

    result = audit(project_dir, tree_root)
    if not result["transcripts"]:
        print(
            f"[capture_transcript] {project_dir} has no transcript records. Skipping.",
            file=sys.stderr,
        )
        return 0

    results_dir = tree_root / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    audit_path = results_dir / f"transcript-audit-{args.tag}.md"
    audit_path.write_text(render(args.tag, tree_root, project_dir, result))
    print(f"[capture_transcript] wrote {audit_path}")

    copied = 0
    total_bytes = 0
    if not args.audit_only:
        raw_dir = results_dir / f"transcript-{args.tag}"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        for src in result["transcripts"]:
            rel = src.relative_to(project_dir)
            dst = raw_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            total_bytes += dst.stat().st_size
        print(
            f"[capture_transcript] copied {copied} transcript file(s), "
            f"{total_bytes / 1048576:.1f} MB, to {raw_dir}"
        )

    outside = len(result["outside"])
    if outside:
        print(
            f"[capture_transcript] NOTE: {outside} out-of-tree path reference(s) — "
            f"see {audit_path.name}"
        )
    else:
        print("[capture_transcript] no out-of-tree path references found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
