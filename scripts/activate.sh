#!/bin/sh
# Load .env from the project root if it exists.
if [ -f "$PIXI_PROJECT_ROOT/.env" ]; then
    set -a
    . "$PIXI_PROJECT_ROOT/.env"
    set +a
fi
