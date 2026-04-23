#!/usr/bin/env bash
# Sync harness files from the template into an existing experiment tree.
#
# Use when a harness fix (fixed_utils.py, program.md, domain.md, pixi config,
# etc.) must reach a live tree mid-experiment. Writes harness files from the
# template's HEAD into the tree and commits them as a single "harness: sync"
# commit. Does not touch the memory surfaces (learnings.md, seed.md, results/,
# plans/) or train.py, which may reflect the agent's own work.
#
# This is a deliberate, auditable action — run it only when you actually want
# the tree to pick up a template change. Routine harness drift is not expected.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

TREE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --tree <path>

Sync harness files from the template ($TEMPLATE_DIR) into an existing tree.
Harness files: program.md, domain.md, fixed_utils.py, pixi.toml, pixi.lock,
               Dockerfile, dprint.json, .gitignore, .gitattributes,
               templates/, example/

NOT synced: train.py, learnings.md, seed.md, results/, plans/, data/, .tree-meta.json.

  --tree <path>   Path to the tree to sync (required)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree) TREE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown flag: $1" >&2; usage; exit 1;;
  esac
done

[[ -n "$TREE" && -d "$TREE/.git" ]] \
  || { echo "--tree must point at a git repo" >&2; exit 1; }

TEMPLATE_SHA_SHORT="$(git -C "$TEMPLATE_DIR" rev-parse --short HEAD)"

# Explicit harness file list. Keep narrow — anything not listed is considered
# tree-owned state and is not overwritten.
HARNESS_FILES=(
  program.md
  domain.md
  fixed_utils.py
  pixi.toml
  pixi.lock
  Dockerfile
  dprint.json
  .gitignore
  .gitattributes
  README.md
)

cd "$TEMPLATE_DIR"
for f in "${HARNESS_FILES[@]}"; do
  if [[ -f "$TEMPLATE_DIR/$f" ]]; then
    mkdir -p "$TREE/$(dirname "$f")"
    cp -p "$TEMPLATE_DIR/$f" "$TREE/$f"
  fi
done

# Directory-level syncs (templates/, example/): mirror contents without
# touching tree-only additions. Use rsync if available; otherwise cp -R.
for d in templates example; do
  if [[ -d "$TEMPLATE_DIR/$d" ]]; then
    mkdir -p "$TREE/$d"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$TEMPLATE_DIR/$d/" "$TREE/$d/"
    else
      rm -rf "$TREE/$d"
      cp -R "$TEMPLATE_DIR/$d" "$TREE/$d"
    fi
  fi
done

cd "$TREE"
if git diff --quiet && git diff --cached --quiet; then
  echo "no harness changes to sync; tree is already at template@${TEMPLATE_SHA_SHORT}"
  exit 0
fi

git add -A
git -c user.name="autoresearch-harness" \
    -c user.email="autoresearch-harness@local" \
    commit -q -m "harness: sync from template@${TEMPLATE_SHA_SHORT}"

echo "synced harness into $TREE at template@${TEMPLATE_SHA_SHORT}"
