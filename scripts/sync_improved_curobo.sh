#!/usr/bin/env bash
# Apply the improved cuRobo patches to a cuRobo source checkout.
#
# Usage:
#   scripts/sync_improved_curobo.sh <curobo-source-root> [--check]
#
#   <curobo-source-root>  Path to a cuRobo checkout whose layout is
#                         src/curobo/... (same layout as the upstream
#                         repository and the bundled Isaac Sim package).
#   --check               Only verify the patches apply; do not modify files.
#
# The patches live in deps/curobo/ and are generated against the cuRobo
# v0.7.5 release source.  If the target checkout is a different version,
# run with --check first and resolve any hunks manually.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$(cd -- "$SCRIPT_DIR/../deps/curobo" && pwd)"

CUROBO_ROOT="${1:-}"
CHECK_ONLY="${2:-}"

if [[ -z "$CUROBO_ROOT" ]]; then
    echo "usage: $0 <curobo-source-root> [--check]" >&2
    exit 2
fi
if [[ ! -d "$CUROBO_ROOT/src/curobo" ]]; then
    echo "error: '$CUROBO_ROOT/src/curobo' does not exist; expected a cuRobo source checkout" >&2
    exit 1
fi
if [[ ! -d "$PATCH_DIR" ]]; then
    echo "error: patch directory not found: $PATCH_DIR" >&2
    exit 1
fi

cd "$CUROBO_ROOT"
applied=0
for patch in "$PATCH_DIR"/*.patch; do
    [[ -f "$patch" ]] || continue
    name="$(basename "$patch")"
    if ! git apply --check "$patch"; then
        echo "error: $name does not apply cleanly to $CUROBO_ROOT (check the cuRobo version)" >&2
        exit 1
    fi
    if [[ "$CHECK_ONLY" == "--check" ]]; then
        echo "OK (check only): $name"
    else
        git apply "$patch"
        echo "applied: $name"
    fi
    applied=$((applied + 1))
done
if [[ "$applied" -eq 0 ]]; then
    echo "error: no patches found in $PATCH_DIR" >&2
    exit 1
fi
echo "sync_improved_curobo: $applied patch(es) handled"
