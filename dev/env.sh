#!/usr/bin/env bash

# Source this file to set TESE project paths in the current shell.
# Usage: source /path/to/TESE/dev/env.sh

# Resolve TESE root from this file location unless already set.
if [ -z "${TESE_ROOT:-}" ]; then
  _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export TESE_ROOT="$(cd "${_ENV_DIR}/.." && pwd)"
  unset _ENV_DIR
fi

export DEV_ROOT="${TESE_ROOT}/dev"
export DATA_ROOT="${DEV_ROOT}/datasets"

echo "TESE_ROOT=${TESE_ROOT}"
echo "DEV_ROOT=${DEV_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
