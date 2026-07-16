#!/usr/bin/env bash
set -euo pipefail

ARAG_COMMIT="a44de6b2216bf6791979c4b6ac4ae106212fa1a6"
ARAG_REPO_URL="${ARAG_REPO_URL:-https://github.com/Ayanami0730/arag.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/baselines/arag"
DESTINATION="${ARAG_DIR:-$REPO_ROOT/arag}"
BOOKRAG_RUNS_DIR="${BOOKRAG_RUNS_DIR:-$REPO_ROOT/BookRAG/runs/financebench_qwen_smoke}"
PREPARE_ONLY=0
DESTINATION_SET=0

usage() {
    cat <<EOF
Usage: $0 [--prepare-only] [destination]

Clone pinned A-RAG, install the local FinanceBench harness, and set it up.

Options:
  --prepare-only  Clone and install the overlay without creating the environment
  -h, --help      Show this help

Environment:
  ARAG_DIR           Default destination (default: $REPO_ROOT/arag)
  ARAG_REPO_URL      Clone URL (default: $ARAG_REPO_URL)
  BOOKRAG_RUNS_DIR   Parsed BookRAG run checked after setup
                     (default: $REPO_ROOT/BookRAG/runs/financebench_qwen_smoke)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-only)
            PREPARE_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ "$DESTINATION_SET" -eq 1 ]]; then
                echo "ERROR: only one destination may be specified" >&2
                exit 2
            fi
            DESTINATION="$1"
            DESTINATION_SET=1
            ;;
    esac
    shift
done

for command in git make; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $command" >&2
        exit 1
    }
done

[[ -d "$OVERLAY_DIR" ]] || {
    echo "ERROR: A-RAG overlay not found at $OVERLAY_DIR" >&2
    exit 1
}

if [[ ! -e "$DESTINATION" ]]; then
    mkdir -p "$(dirname "$DESTINATION")"
    echo "Cloning A-RAG into $DESTINATION"
    git clone "$ARAG_REPO_URL" "$DESTINATION"
    git -C "$DESTINATION" checkout "$ARAG_COMMIT"
elif [[ ! -d "$DESTINATION/.git" ]]; then
    echo "ERROR: destination exists but is not a Git checkout: $DESTINATION" >&2
    exit 1
fi

current_commit="$(git -C "$DESTINATION" rev-parse HEAD)"
if [[ "$current_commit" != "$ARAG_COMMIT" ]]; then
    echo "ERROR: $DESTINATION is at $current_commit" >&2
    echo "Expected pinned A-RAG commit $ARAG_COMMIT; refusing to change an existing checkout." >&2
    exit 1
fi

echo "Installing the A-RAG harness"
cp -R "$OVERLAY_DIR"/. "$DESTINATION"/

if [[ ! -e "$DESTINATION/.env" && ! -L "$DESTINATION/.env" ]]; then
    if [[ -f "$REPO_ROOT/.env" ]]; then
        ln -s "$REPO_ROOT/.env" "$DESTINATION/.env"
        echo "Linked $DESTINATION/.env to the repository .env"
    else
        echo "WARNING: $REPO_ROOT/.env is missing; create it before answering" >&2
    fi
fi

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    echo "A-RAG prepared at $DESTINATION"
    exit 0
fi

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: required command not found: uv" >&2
    exit 1
}

make -C "$DESTINATION" setup
make -C "$DESTINATION" check \
    FINANCEBENCH_DIR="$REPO_ROOT" \
    BOOKRAG_RUNS_DIR="$BOOKRAG_RUNS_DIR"

echo "A-RAG is ready at $DESTINATION"
echo "Run: make -C \"$DESTINATION\" smoke FINANCEBENCH_DIR=\"$REPO_ROOT\" BOOKRAG_RUNS_DIR=\"$BOOKRAG_RUNS_DIR\""
