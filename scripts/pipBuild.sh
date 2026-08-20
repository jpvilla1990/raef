#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(dirname "${BASH_SOURCE[0]}")"

export UV_PUBLISH_TOKEN=${UV_PUBLISH_TOKEN:-token}

(
    cd "${SCRIPT_PATH}"
    rm -rf README.md
    rm -rf src
    rm -rf dist
    rm -rf uv.lock
    ln -s ../README.md README.md
    ln -s ../src src

    uv build
    uv publish

    rm -rf README.md
    rm -rf src
    rm -rf dist
    rm -rf uv.lock
)