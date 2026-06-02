#!/bin/bash
set -e
cd "$(dirname "$0")/cache-agent"
go build -o "../cache-agent" .
echo "Built cache-agent binary"
