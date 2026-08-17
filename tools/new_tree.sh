#!/usr/bin/env bash
# Create an isolated experiment tree from this template.
#
# A tree is a fresh git repo with exactly one commit, so the agent cannot see
# prior sessions through `git log --all`, `git tag -l`, or an inherited
# learnings.md. That isolation is the point: each tree is an independent
# sample of what the method finds, not a continuation of the last run.
#
# Usage:
#   tools/new_tree.sh <target-dir> [--seed-md FILE] [--no-domain]
#
#   --seed-md FILE   install FILE as the tree's seed.md (the human's brief)
#   --no-domain      omit domain.md entirely — an unguarded run with no
#                    domain context and no constraints

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TARGET=""
SEED_MD=""
NO_DOMAIN=0

usage() {
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-md)
      [[ $# -ge 2 ]] || { echo "--seed-md needs a file" >&2; exit 1; }
      SEED_MD="$2"; shift 2 ;;
    --no-domain)
      NO_DOMAIN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    -*)
      echo "unknown flag: $1" >&2; exit 1 ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "target dir given twice: $TARGET and $1" >&2
        exit 1
      fi
      TARGET="$1"; shift ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  usage >&2
  exit 1
fi
if [[ -e "$TARGET" ]]; then
  echo "refusing to overwrite existing path: $TARGET" >&2
  exit 1
fi
if [[ -n "$SEED_MD" ]]; then
  if [[ ! -f "$SEED_MD" ]]; then
    echo "no such file: $SEED_MD" >&2
    exit 1
  fi
  # Absolutize before we start creating directories and moving around.
  SEED_MD="$(cd -- "$(dirname -- "$SEED_MD")" && pwd)/$(basename -- "$SEED_MD")"
fi

mkdir -p "$TARGET"
TARGET="$(cd -- "$TARGET" && pwd)"

# Copy every tracked file except the tree-creation tooling itself.
git -C "$TEMPLATE_DIR" ls-files -z \
  | grep -zv '^tools/new_tree\.sh$' \
  | tar -C "$TEMPLATE_DIR" --null -T - -cf - \
  | tar -C "$TARGET" -xf -

# A tree starts with no accumulated state: no prior findings, no prior
# results, no prior plans. Anything inherited here would contaminate the run.
: > "$TARGET/learnings.md"
: > "$TARGET/seed.md"
rm -rf "$TARGET/results" "$TARGET/plans"
mkdir -p "$TARGET/results" "$TARGET/plans"
touch "$TARGET/results/.gitkeep" "$TARGET/plans/.gitkeep"

if [[ -n "$SEED_MD" ]]; then
  cp "$SEED_MD" "$TARGET/seed.md"
fi
if [[ "$NO_DOMAIN" -eq 1 ]]; then
  rm -f "$TARGET/domain.md"
fi

# Point the tree at the template's dataset via a relative symlink, so a
# checkout of any historical experiment tag resolves its data paths unchanged.
rm -rf "$TARGET/data"
ln -s "$(realpath --relative-to="$TARGET" "$TEMPLATE_DIR/data")" "$TARGET/data"

TEMPLATE_SHA="$(git -C "$TEMPLATE_DIR" rev-parse --short HEAD)"
git -C "$TARGET" init -q
git -C "$TARGET" add -A
git -C "$TARGET" commit -q -m "initial scaffold (from template@${TEMPLATE_SHA})"

if [[ "$NO_DOMAIN" -eq 1 ]]; then domain_state="omitted (--no-domain)"; else domain_state="present"; fi
if [[ -n "$SEED_MD" ]]; then seed_state="$SEED_MD"; else seed_state="empty"; fi

echo "tree ready: $TARGET"
echo "  template  : ${TEMPLATE_DIR} @ ${TEMPLATE_SHA}"
echo "  domain.md : ${domain_state}"
echo "  seed.md   : ${seed_state}"
echo
echo "  cd $TARGET && claude"
