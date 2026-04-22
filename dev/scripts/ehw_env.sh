#!/usr/bin/env bash

if [ -z "${TESE_ROOT:-}" ]; then
  _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export TESE_ROOT="$(cd "${_ENV_DIR}/.." && pwd)"
  unset _ENV_DIR
fi

export DEV_ROOT="${TESE_ROOT}/dev"
export DATA_ROOT="${DEV_ROOT}/datasets"

export EHW_RAW_ROOT="${DATA_ROOT}/raw/EHWGesture"
export EHW_EVENT_ROOT="${EHW_RAW_ROOT}/DataEvent"
export EHW_EVT_ROOT="${DATA_ROOT}/evt_og/EHWGesture"
export EHW_TS_ROOT="${DATA_ROOT}/timesformer/EHWGesture"

export PYTHONPATH="${DEV_ROOT}/TimeSformer${PYTHONPATH:+:${PYTHONPATH}}"

echo "TESE_ROOT=${TESE_ROOT}"
echo "DEV_ROOT=${DEV_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "EHW_RAW_ROOT=${EHW_RAW_ROOT}"
echo "EHW_EVENT_ROOT=${EHW_EVENT_ROOT}"
echo "EHW_EVT_ROOT=${EHW_EVT_ROOT}"
echo "EHW_TS_ROOT=${EHW_TS_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
