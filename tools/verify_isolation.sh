#!/usr/bin/env bash
# Verify that an experiment tree is isolated from prior runs.
#
# A tree is meant to be an independent sample of what the method finds. That
# claim is only as good as the channels it closes, so check them rather than
# assume them. Run this from inside a tree BEFORE starting a session, and
# keep the output — it is the evidence that the run was independent.
#
# Usage:
#   tools/verify_isolation.sh [tree-dir]     (default: current directory)
#
# Exits non-zero if any check fails.

set -uo pipefail

TREE="${1:-$PWD}"
TREE="$(cd -- "$TREE" && pwd)"
FAILED=0

pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }

echo "isolation check: $TREE"
echo

# --- 1. the tree's own git history reveals nothing -------------------------

if [[ ! -d "$TREE/.git" ]]; then
  fail "no .git in the tree — git would walk up into a parent repo's history"
else
  pass "tree has its own .git (git cannot walk up past it)"
fi

commits="$(git -C "$TREE" rev-list --count --all 2>/dev/null || echo -1)"
if [[ "$commits" == "1" ]]; then
  pass "history is a single commit"
else
  fail "expected 1 commit, found ${commits} — a session may already have run here"
fi

tags="$(git -C "$TREE" tag -l 2>/dev/null | wc -l)"
if [[ "$tags" -eq 0 ]]; then
  pass "no tags"
else
  fail "found ${tags} tag(s) — prior experiments are addressable from here"
fi

branches="$(git -C "$TREE" branch -a --format='%(refname)' 2>/dev/null | wc -l)"
if [[ "$branches" -le 1 ]]; then
  pass "no extra branches"
else
  fail "found ${branches} branches — expected 1"
fi

remotes="$(git -C "$TREE" remote 2>/dev/null | wc -l)"
if [[ "$remotes" -eq 0 ]]; then
  pass "no git remotes (nothing to fetch history from)"
else
  fail "found ${remotes} remote(s) — history is reachable via fetch"
fi

# --- 2. no inherited findings ----------------------------------------------

if [[ -f "$TREE/learnings.md" && ! -s "$TREE/learnings.md" ]]; then
  pass "learnings.md is empty"
elif [[ ! -f "$TREE/learnings.md" ]]; then
  fail "learnings.md is missing"
else
  fail "learnings.md is non-empty ($(wc -c <"$TREE/learnings.md") bytes) — findings inherited"
fi

stray="$(find "$TREE/results" "$TREE/plans" -type f ! -name '.gitkeep' 2>/dev/null | wc -l)"
if [[ "$stray" -eq 0 ]]; then
  pass "results/ and plans/ are empty"
else
  fail "${stray} pre-existing file(s) under results/ or plans/"
fi

# --- 3. the dataset does not lead back into a repository -------------------
#
# The subtle one: if data/ resolves inside a git repo, `git -C data log` and
# `git -C data tag -l` expose that repo's entire history from inside the tree.

if [[ -e "$TREE/data" ]]; then
  data_real="$(readlink -f "$TREE/data")"
  if data_repo="$(git -C "$data_real" rev-parse --show-toplevel 2>/dev/null)"; then
    fail "data/ resolves into a git repo: ${data_repo}"
    printf '        (%s commits, %s tags reachable via `git -C data log`)\n' \
      "$(git -C "$data_real" rev-list --count --all 2>/dev/null)" \
      "$(git -C "$data_real" tag -l 2>/dev/null | wc -l)"
  else
    pass "data/ resolves outside any git repo (${data_real})"
  fi
else
  fail "data/ is missing — train.py will not run"
fi

# --- 4. no ambient instructions ---------------------------------------------
#
# Claude Code loads CLAUDE.md from the working directory and every ancestor,
# so one dropped anywhere above the tree silently enters the session.

found_claude_md=""
probe="$TREE"
while :; do
  [[ -f "$probe/CLAUDE.md" ]] && found_claude_md+="${probe}/CLAUDE.md "
  [[ "$probe" == "/" ]] && break
  probe="$(dirname "$probe")"
done
[[ -f "$HOME/.claude/CLAUDE.md" ]] && found_claude_md+="$HOME/.claude/CLAUDE.md "

if [[ -z "$found_claude_md" ]]; then
  pass "no CLAUDE.md in the tree or any ancestor"
else
  fail "CLAUDE.md present, will be loaded into the session: ${found_claude_md}"
fi

# --- 5. nothing experiment-shaped sits next to the tree --------------------
#
# Git isolation does not stop `ls ..`. A sibling run is readable if it exists.

parent="$(dirname "$TREE")"
siblings=0
while IFS= read -r d; do
  [[ "$d" == "$TREE" ]] && continue
  if [[ -e "$d/train.py" || -e "$d/program.md" || -e "$d/learnings.md" ]]; then
    siblings=$((siblings + 1))
    printf '        sibling run: %s\n' "$d"
  fi
done < <(find "$parent" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

if [[ "$siblings" -eq 0 ]]; then
  pass "no sibling experiment trees in ${parent}"
else
  fail "${siblings} sibling run(s) readable one directory up"
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "PASS — tree is isolated"
else
  echo "FAIL — see above; this run would not be an independent sample"
fi
exit "$FAILED"
