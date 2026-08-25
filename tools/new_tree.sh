#!/usr/bin/env bash
# Create an isolated experiment tree from this template.
#
# A tree is a fresh git repo with exactly one commit, so the agent cannot see
# prior sessions through `git log --all` or an inherited learnings.md. That
# isolation is the point: each tree is an independent sample of what the
# method finds, not a continuation of the last run.
#
# Usage:
#   tools/new_tree.sh <target-dir> [--data DIR]
#
#   --data DIR       dataset directory to link as the tree's data/
#                    (default: this template's data/). Point it somewhere
#                    outside any git repo: if data/ resolves inside one,
#                    `git -C data log` exposes that repo's whole history
#                    from inside the tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TARGET=""
DATA_DIR=""

# Print the header comment block, whatever length it happens to be. A fixed
# line range here goes stale silently the first time the header changes.
usage() {
    awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            [[ $# -ge 2 ]] || { echo "--data needs a directory" >&2; exit 1; }
        DATA_DIR="$2"; shift 2 ;;
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

# Session transcripts are committed into the tree through a git-LFS filter
# declared in .gitattributes. Without git-lfs on PATH that filter is inert and
# the transcripts land as ordinary blobs — recoverable, but it silently
# changes how the tree stores its own evidence. Fail here instead.
if ! git lfs version &>/dev/null; then
    echo "git-lfs is not installed, but .gitattributes routes session" >&2
    echo "transcripts through it. Install git-lfs and re-run." >&2
    exit 1
fi

if [[ -n "$DATA_DIR" ]]; then
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "no such directory: $DATA_DIR" >&2
        exit 1
    fi
    DATA_DIR="$(cd -- "$DATA_DIR" && pwd)"
else
    DATA_DIR="$TEMPLATE_DIR/data"
fi

mkdir -p "$TARGET"
TARGET="$(cd -- "$TARGET" && pwd)"

git -C "$TEMPLATE_DIR" ls-files -z \
| grep -zvE '^(tools/new_tree\.sh|README\.md)$' \
| tar -C "$TEMPLATE_DIR" --null -T - -cf - \
| tar -C "$TARGET" -xf -

# A tree starts with no accumulated state: no prior findings, no prior
# results, no prior plans. Anything inherited here would contaminate the run.
: > "$TARGET/learnings.md"
rm -rf "$TARGET/results" "$TARGET/plans"
mkdir -p "$TARGET/results" "$TARGET/plans"
touch "$TARGET/results/.gitkeep" "$TARGET/plans/.gitkeep"

# Link the dataset in, so a checkout of any historical experiment commit
# resolves its data paths unchanged. Warn loudly if it lands inside a git
# repo — that would expose the repo's history via `git -C data log`.
rm -rf "$TARGET/data"
ln -s "$DATA_DIR" "$TARGET/data"
if data_repo="$(git -C "$(readlink -f "$TARGET/data")" rev-parse --show-toplevel 2>/dev/null)"; then
    echo "WARNING: data/ resolves inside a git repo (${data_repo})." >&2
    echo "         Prior experiment history is readable from inside the tree." >&2
    echo "         Pass --data with a directory outside any repo." >&2
fi

TEMPLATE_SHA="$(git -C "$TEMPLATE_DIR" rev-parse HEAD)"
printf '%s\n' "$TEMPLATE_SHA" > "$TARGET/TEMPLATE_COMMIT"
# -b main, not the git default: the protocol runs the whole session on `main`,
# and its instructions say so by name.
git -C "$TARGET" init -q -b main
# --local, not global: this writes only the tree's .git/config, so creating a
# tree never reconfigures the operator's machine.
git -C "$TARGET" lfs install --local &>/dev/null
git -C "$TARGET" add -A
git -C "$TARGET" commit -q -m "initial scaffold (from template@${TEMPLATE_SHA:0:12})"

echo "tree ready: $TARGET"
echo "  template  : ${TEMPLATE_DIR} @ ${TEMPLATE_SHA}"
echo "  data      : ${DATA_DIR}"
echo
echo "  cd $TARGET && claude"
