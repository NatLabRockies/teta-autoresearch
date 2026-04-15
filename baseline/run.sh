#!/usr/bin/env bash
# Convenience wrapper: activates the pixi environment and runs the search.
# All arguments are forwarded to search.py.
#
# Usage:
#   ./run.sh --tag apr15 --n-trials 200
#   ./run.sh --tag apr15 --n-trials 200 --sampler cmaes --budget 300
#   ./run.sh --help

set -euo pipefail
cd "$(dirname "$0")"
pixi run python search.py "$@"
