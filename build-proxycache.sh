#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/go-proxycache"
GO_BIN="/mnt/c/Program Files/Go/bin/go.exe"
if [ -x "$GO_BIN" ]; then
    "$GO_BIN" build
else
    go build
fi
mv proxycache.exe "$SCRIPT_DIR/proxycache.exe"
echo "Built proxycache binary"
