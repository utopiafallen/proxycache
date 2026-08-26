#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
GO_BIN="/mnt/c/Program Files/Go/bin/go.exe"
if [ -x "$GO_BIN" ]; then
    "$GO_BIN" build
else
    go build
fi
echo "Built proxycache binary"
