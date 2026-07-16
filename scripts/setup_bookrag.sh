#!/usr/bin/env bash
set -euo pipefail

BOOKRAG_COMMIT="113298f919c701d07807ceccb96ed3c18d348117"
BOOKRAG_REPO_URL="${BOOKRAG_REPO_URL:-https://github.com/sam234990/BookRAG.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/baselines/bookrag"
DESTINATION="${BOOKRAG_DIR:-$REPO_ROOT/BookRAG}"
PREPARE_ONLY=0
DESTINATION_SET=0

usage() {
    cat <<EOF
Usage: $0 [--prepare-only] [destination]

Clone pinned BookRAG, install the local FinanceBench harness, and set it up.

Options:
  --prepare-only  Clone and install the overlay without creating the environment
  -h, --help      Show this help

Environment:
  BOOKRAG_DIR       Default destination (default: $REPO_ROOT/BookRAG)
  BOOKRAG_REPO_URL  Clone URL (default: $BOOKRAG_REPO_URL)
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
    echo "ERROR: BookRAG overlay not found at $OVERLAY_DIR" >&2
    exit 1
}

if [[ ! -e "$DESTINATION" ]]; then
    mkdir -p "$(dirname "$DESTINATION")"
    echo "Cloning BookRAG into $DESTINATION"
    git clone "$BOOKRAG_REPO_URL" "$DESTINATION"
    git -C "$DESTINATION" checkout "$BOOKRAG_COMMIT"
elif [[ ! -d "$DESTINATION/.git" ]]; then
    echo "ERROR: destination exists but is not a Git checkout: $DESTINATION" >&2
    exit 1
fi

current_commit="$(git -C "$DESTINATION" rev-parse HEAD)"
if [[ "$current_commit" != "$BOOKRAG_COMMIT" ]]; then
    echo "ERROR: $DESTINATION is at $current_commit" >&2
    echo "Expected pinned BookRAG commit $BOOKRAG_COMMIT; refusing to change an existing checkout." >&2
    exit 1
fi

echo "Installing the BookRAG harness"
cp -R "$OVERLAY_DIR"/. "$DESTINATION"/

if [[ ! -e "$DESTINATION/.env" && ! -L "$DESTINATION/.env" ]]; then
    if [[ -f "$REPO_ROOT/.env" ]]; then
        ln -s "$REPO_ROOT/.env" "$DESTINATION/.env"
        echo "Linked $DESTINATION/.env to the repository .env"
    else
        echo "WARNING: $REPO_ROOT/.env is missing; create it before running BookRAG" >&2
    fi
fi

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    echo "BookRAG prepared at $DESTINATION"
    exit 0
fi

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: required command not found: uv" >&2
    exit 1
}

make -C "$DESTINATION" setup
make -C "$DESTINATION" check FINANCEBENCH_DIR="$REPO_ROOT"

echo "BookRAG is ready at $DESTINATION"
echo "Run: make -C \"$DESTINATION\" smoke FINANCEBENCH_DIR=\"$REPO_ROOT\""
