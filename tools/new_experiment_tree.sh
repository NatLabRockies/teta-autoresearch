#!/usr/bin/env bash
# Create an isolated experiment tree from the routee-autoresearch harness.
#
# Each tree is a separate git repo in its own sibling directory. The tree has
# exactly one prior commit (the scaffold) so `git log --all`, `git tag -l`,
# and `git branch -a` reveal nothing from the template's history or from
# sibling trees. Bias channels (learnings.md, prior tags, session branches,
# committed results/plans) are wiped; harness files (program.md, domain.md,
# fixed_utils.py, train.py scaffold, pixi config, templates) are preserved.
#
# See tools/README-trees.md for the surrounding workflow.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

NAME=""
POWERTRAIN=""
SEED_MD=""
SEED_LEARNINGS=""
TARGET_DIR=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --name <tree> --powertrain <bev|ice|phev> [options]

Create an isolated experiment tree.

Required:
  --name <tree>             Tree name (used for dir and registry entry)
  --powertrain <pw>         bev | ice | phev — sets POWERTRAIN in train.py

Optional:
  --seed-md <file>          File to use as seed.md (default: empty)
  --seed-learnings <file>   File to use as learnings.md (default: empty)
  --target-dir <path>       Where to create the tree
                            (default: ~/repos/routee-autoresearch-trees/<name>)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)            NAME="$2"; shift 2;;
    --powertrain)      POWERTRAIN="$2"; shift 2;;
    --seed-md)         SEED_MD="$2"; shift 2;;
    --seed-learnings) SEED_LEARNINGS="$2"; shift 2;;
    --target-dir)      TARGET_DIR="$2"; shift 2;;
    -h|--help)         usage; exit 0;;
    *) echo "unknown flag: $1" >&2; usage; exit 1;;
  esac
done

[[ -n "$NAME" ]] || { echo "--name is required" >&2; exit 1; }
[[ "$POWERTRAIN" =~ ^(bev|ice|phev)$ ]] \
  || { echo "--powertrain must be bev|ice|phev" >&2; exit 1; }

if [[ -z "$TARGET_DIR" ]]; then
  TARGET_DIR="${HOME}/repos/routee-autoresearch-trees/${NAME}"
fi

if [[ -e "$TARGET_DIR" ]]; then
  echo "target already exists: $TARGET_DIR" >&2; exit 1
fi

if [[ -n "$SEED_MD" && ! -f "$SEED_MD" ]]; then
  echo "--seed-md file not found: $SEED_MD" >&2; exit 1
fi
if [[ -n "$SEED_LEARNINGS" && ! -f "$SEED_LEARNINGS" ]]; then
  echo "--seed-learnings file not found: $SEED_LEARNINGS" >&2; exit 1
fi

# Resolve seed paths to absolute before we cd around.
[[ -n "$SEED_MD" ]] && SEED_MD="$(cd "$(dirname "$SEED_MD")" && pwd)/$(basename "$SEED_MD")"
[[ -n "$SEED_LEARNINGS" ]] && SEED_LEARNINGS="$(cd "$(dirname "$SEED_LEARNINGS")" && pwd)/$(basename "$SEED_LEARNINGS")"

TEMPLATE_SHA="$(git -C "$TEMPLATE_DIR" rev-parse HEAD)"
TEMPLATE_SHA_SHORT="$(git -C "$TEMPLATE_DIR" rev-parse --short HEAD)"

mkdir -p "$TARGET_DIR"

# Paths we skip when copying — they're either wiped (memory), replaced
# (results/plans skeletons), or substituted (data → symlink).
#  - learnings.md, seed.md: wiped/seeded below
#  - results/*, plans/*: skeletons recreated below
#  - data/*: symlinked below (saves disk; data is read-only)
EXCLUDE_REGEX='^(learnings\.md|seed\.md|results/|plans/|data/)'

cd "$TEMPLATE_DIR"
while IFS= read -r f; do
  [[ "$f" =~ $EXCLUDE_REGEX ]] && continue
  mkdir -p "$TARGET_DIR/$(dirname "$f")"
  cp -p "$f" "$TARGET_DIR/$f"
done < <(git ls-files)

ln -s "$TEMPLATE_DIR/data" "$TARGET_DIR/data"

if [[ -n "$SEED_MD" ]]; then
  cp "$SEED_MD" "$TARGET_DIR/seed.md"
else
  : > "$TARGET_DIR/seed.md"
fi

if [[ -n "$SEED_LEARNINGS" ]]; then
  cp "$SEED_LEARNINGS" "$TARGET_DIR/learnings.md"
else
  : > "$TARGET_DIR/learnings.md"
fi

# results/ skeleton — mirror the template's partition layout so the agent's
# "initialize results files" step in program.md has somewhere to write.
mkdir -p "$TARGET_DIR/results/bev" "$TARGET_DIR/results/ice" "$TARGET_DIR/results/phev"
: > "$TARGET_DIR/results/.gitkeep"
: > "$TARGET_DIR/results/bev/.gitkeep"
: > "$TARGET_DIR/results/ice/.gitkeep"
: > "$TARGET_DIR/results/phev/.gitkeep"

# Set POWERTRAIN selector in train.py.
python3 - "$TARGET_DIR/train.py" "$POWERTRAIN" <<'PY'
import re, pathlib, sys
path, pw = pathlib.Path(sys.argv[1]), sys.argv[2]
src = path.read_text()
new, n = re.subn(r'^POWERTRAIN\s*=.*$', f'POWERTRAIN = "{pw}"  # session selector — the only vehicle-related line to change', src, count=1, flags=re.M)
if n != 1:
    sys.exit(f"POWERTRAIN assignment not found in {path}")
path.write_text(new)
PY

NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SEED_MD_REF="${SEED_MD:-<empty>}"
SEED_LEARN_REF="${SEED_LEARNINGS:-<empty>}"

python3 - "$TARGET_DIR/.tree-meta.json" \
  "$NAME" "$TEMPLATE_SHA" "$TEMPLATE_SHA_SHORT" "$POWERTRAIN" \
  "$NOW_UTC" "$SEED_MD_REF" "$SEED_LEARN_REF" <<'PY'
import json, sys
(out, name, sha, sha_short, pw, now, seed_md, seed_learn) = sys.argv[1:9]
with open(out, "w") as f:
    json.dump({
        "name": name,
        "template_sha": sha,
        "template_sha_short": sha_short,
        "powertrain": pw,
        "created_utc": now,
        "seed_md_source": seed_md,
        "seed_learnings_source": seed_learn,
    }, f, indent=2)
    f.write("\n")
PY

cd "$TARGET_DIR"
git init -q -b main
git config advice.detachedHead false
git config advice.addIgnoredFile false
git add -A
git -c user.name="autoresearch-harness" \
    -c user.email="autoresearch-harness@local" \
    commit -q -m "initial scaffold (from template@${TEMPLATE_SHA_SHORT})"

REGISTRY_DIR="${HOME}/repos/routee-autoresearch-trees"
mkdir -p "$REGISTRY_DIR"
REGISTRY="${REGISTRY_DIR}/registry.jsonl"
python3 - "$REGISTRY" \
  "$NAME" "$TARGET_DIR" "$TEMPLATE_SHA" "$POWERTRAIN" \
  "$NOW_UTC" "$SEED_MD_REF" "$SEED_LEARN_REF" <<'PY'
import json, sys
(reg, name, target, sha, pw, now, seed_md, seed_learn) = sys.argv[1:9]
with open(reg, "a") as f:
    f.write(json.dumps({
        "name": name,
        "target_dir": target,
        "template_sha": sha,
        "powertrain": pw,
        "created_utc": now,
        "seed_md_source": seed_md,
        "seed_learnings_source": seed_learn,
    }) + "\n")
PY

cat <<EOF

created tree:
  name:       $NAME
  path:       $TARGET_DIR
  template:   ${TEMPLATE_SHA_SHORT}
  powertrain: $POWERTRAIN
  seed.md:    ${SEED_MD_REF}
  learnings:  ${SEED_LEARN_REF}

registry:     ${REGISTRY}

start an agent from within the tree dir.
EOF
