#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-baseline}
# Allow METHOD as alias
if [ -n "${METHOD:-}" ]; then
  MODE="$METHOD"
fi
echo "Mode: $MODE"

case "$MODE" in
  baseline)
    python scripts/baseline_seasonality.py
    ;;
  geometric)
    python scripts/baseline_geometric.py
    ;;
  simple)
    python scripts/baseline_simple.py
    ;;
  ridge)
    python scripts/baseline_ridge.py
    ;;
  *)
    echo "Unknown MODE: $MODE"
    echo "Valid: baseline | geometric | simple | ridge"
    exit 1
    ;;
esac



