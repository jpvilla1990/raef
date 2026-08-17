#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(dirname "${BASH_SOURCE[0]}")"

export UV_PUBLISH_TOKEN=${UV_PUBLISH_TOKEN:-pypi-AgEIcHlwaS5vcmcCJGI2OTFkMTQzLTJiMTItNDNiYS05NzhjLTc1MjM0OWJlMWE3YwACKlszLCIzOThhYzAzOS1iYTk0LTQ1NTYtYWM2NC1mMjNhZmQyNGViMDMiXQAABiBcowognU_5aaN79hfUSWBSu9UG89V1vH67IY1j3Xijuw}

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