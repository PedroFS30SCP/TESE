#!/usr/bin/env bash
set -euo pipefail

MODE="launch"
if [[ "${1:-}" == "__monitor__" ]]; then
  MODE="monitor"
  shift
fi

TITLE=""
TOPIC=""
WORKDIR=""
LOG_PATH=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
METRICS_ROOT=""
NOTIFY_EVERY=5
ACC_COLUMN="val_acc"
LOSS_COLUMN="val_loss_total"
PID=""
NICE_LEVEL=10

usage() {
  cat <<'EOF'
Usage:
  run_training_with_ntfy.sh \
    --title "Job title" \
    --topic "ntfy_topic" \
    --workdir "/abs/workdir" \
    --log "/abs/path/to/train.log" \
    [--python-bin "/abs/python"] \
    [--metrics-root "/abs/output_root"] \
    [--notify-every 5] \
    [--acc-column val_acc] \
    [--loss-column val_loss_total] \
    [--nice-level 10] \
    -- command arg1 arg2 ...

The script starts the command with nohup, writes to exactly one log file,
and sends ntfy notifications at start, every N epochs (if metrics CSV exists),
and on finish/failure.
EOF
}

notify() {
  local title="$1"
  local body="$2"
  curl -fsS -H "Title: ${title}" -d "${body}" "https://ntfy.sh/${TOPIC}" >/dev/null 2>&1 || true
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --title)
        TITLE="$2"
        shift 2
        ;;
      --topic)
        TOPIC="$2"
        shift 2
        ;;
      --workdir)
        WORKDIR="$2"
        shift 2
        ;;
      --log)
        LOG_PATH="$2"
        shift 2
        ;;
      --python-bin)
        PYTHON_BIN="$2"
        shift 2
        ;;
      --metrics-root)
        METRICS_ROOT="$2"
        shift 2
        ;;
      --notify-every)
        NOTIFY_EVERY="$2"
        shift 2
        ;;
      --acc-column)
        ACC_COLUMN="$2"
        shift 2
        ;;
      --loss-column)
        LOSS_COLUMN="$2"
        shift 2
        ;;
      --pid)
        PID="$2"
        shift 2
        ;;
      --nice-level)
        NICE_LEVEL="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
  REMAINING_ARGS=("$@")
}

latest_metrics_file() {
  if [[ -z "${METRICS_ROOT}" ]]; then
    return 1
  fi
  find "${METRICS_ROOT}" -path "*/train_log/version_*/metrics.csv" | sort -V | tail -n 1
}

monitor_loop() {
  local last_sent_epoch=0

  while kill -0 "${PID}" 2>/dev/null; do
    local metrics_file=""
    metrics_file="$(latest_metrics_file || true)"
    if [[ -n "${metrics_file}" && -f "${metrics_file}" ]]; then
      local msg=""
      msg="$("${PYTHON_BIN}" - <<'PY' "${metrics_file}" "${ACC_COLUMN}" "${LOSS_COLUMN}" "${NOTIFY_EVERY}" "${last_sent_epoch}"
import sys
import pandas as pd

path, acc_col, loss_col, every_s, last_sent_s = sys.argv[1:]
every = int(every_s)
last_sent = int(last_sent_s)

try:
    df = pd.read_csv(path)
except Exception:
    print("")
    raise SystemExit(0)

if "epoch" not in df.columns:
    print("")
    raise SystemExit(0)

df = df[df["epoch"].notna()].copy()
if df.empty:
    print("")
    raise SystemExit(0)

mask = pd.Series(False, index=df.index)
if acc_col in df.columns:
    mask = mask | df[acc_col].notna()
if loss_col in df.columns:
    mask = mask | df[loss_col].notna()
if mask.any():
    df = df[mask]
if df.empty:
    print("")
    raise SystemExit(0)

row = df.iloc[-1]
epoch = int(row["epoch"]) + 1
if epoch % every != 0 or epoch <= last_sent:
    print("")
    raise SystemExit(0)

acc = row[acc_col] if acc_col in row.index and pd.notna(row[acc_col]) else float("nan")
loss = row[loss_col] if loss_col in row.index and pd.notna(row[loss_col]) else float("nan")
print(f"{epoch}|{acc}|{loss}")
PY
)"
      if [[ -n "${msg}" ]]; then
        IFS="|" read -r epoch acc loss <<< "${msg}"
        last_sent_epoch="${epoch}"
        notify "${TITLE} epoch ${epoch}" "Epoch ${epoch} on $(hostname)
${ACC_COLUMN}=${acc}
${LOSS_COLUMN}=${loss}
log=${LOG_PATH}"
      fi
    fi
    sleep 60
  done

  local status=0
  if wait "${PID}" 2>/dev/null; then
    status=0
  else
    status=$?
  fi

  local metrics_file=""
  metrics_file="$(latest_metrics_file || true)"
  local final_epoch="unknown"
  local final_acc="unknown"
  local final_loss="unknown"

  if [[ -n "${metrics_file}" && -f "${metrics_file}" ]]; then
    local final_msg=""
    final_msg="$("${PYTHON_BIN}" - <<'PY' "${metrics_file}" "${ACC_COLUMN}" "${LOSS_COLUMN}"
import sys
import pandas as pd

path, acc_col, loss_col = sys.argv[1:]
try:
    df = pd.read_csv(path)
except Exception:
    print("unknown|unknown|unknown")
    raise SystemExit(0)

if "epoch" not in df.columns:
    print("unknown|unknown|unknown")
    raise SystemExit(0)

df = df[df["epoch"].notna()].copy()
if df.empty:
    print("unknown|unknown|unknown")
    raise SystemExit(0)

mask = pd.Series(False, index=df.index)
if acc_col in df.columns:
    mask = mask | df[acc_col].notna()
if loss_col in df.columns:
    mask = mask | df[loss_col].notna()
if mask.any():
    df = df[mask]
if df.empty:
    print("unknown|unknown|unknown")
    raise SystemExit(0)

row = df.iloc[-1]
epoch = int(row["epoch"]) + 1
acc = row[acc_col] if acc_col in row.index and pd.notna(row[acc_col]) else "unknown"
loss = row[loss_col] if loss_col in row.index and pd.notna(row[loss_col]) else "unknown"
print(f"{epoch}|{acc}|{loss}")
PY
)"
    IFS="|" read -r final_epoch final_acc final_loss <<< "${final_msg}"
  fi

  if [[ "${status}" -eq 0 ]]; then
    notify "${TITLE} finished" "Finished on $(hostname) at $(date)
epoch=${final_epoch}
${ACC_COLUMN}=${final_acc}
${LOSS_COLUMN}=${final_loss}
log=${LOG_PATH}"
  else
    local err_tail=""
    err_tail="$(tail -n 30 "${LOG_PATH}" 2>/dev/null || true)"
    notify "${TITLE} failed" "Failed on $(hostname) at $(date)
status=${status}
epoch=${final_epoch}
${ACC_COLUMN}=${final_acc}
${LOSS_COLUMN}=${final_loss}

Last log lines:
${err_tail}"
  fi
}

parse_args "$@"

if [[ "${MODE}" == "monitor" ]]; then
  if [[ -z "${PID}" ]]; then
    echo "Missing --pid for monitor mode" >&2
    exit 1
  fi
  monitor_loop
  exit 0
fi

if [[ -z "${TITLE}" || -z "${TOPIC}" || -z "${WORKDIR}" || -z "${LOG_PATH}" ]]; then
  usage >&2
  exit 1
fi

if [[ ${#REMAINING_ARGS[@]} -eq 0 ]]; then
  echo "Missing command after --" >&2
  usage >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_PATH}")"
cd "${WORKDIR}"

notify "${TITLE} started" "Started on $(hostname) at $(date)
workdir=${WORKDIR}
log=${LOG_PATH}"

nohup nice -n "${NICE_LEVEL}" "${REMAINING_ARGS[@]}" >> "${LOG_PATH}" 2>&1 < /dev/null &
TRAIN_PID=$!

nohup "$0" __monitor__ \
  --title "${TITLE}" \
  --topic "${TOPIC}" \
  --workdir "${WORKDIR}" \
  --log "${LOG_PATH}" \
  --python-bin "${PYTHON_BIN}" \
  --metrics-root "${METRICS_ROOT}" \
  --notify-every "${NOTIFY_EVERY}" \
  --acc-column "${ACC_COLUMN}" \
  --loss-column "${LOSS_COLUMN}" \
  --pid "${TRAIN_PID}" \
  > /dev/null 2>&1 < /dev/null &

echo "Started ${TITLE}"
echo "PID=${TRAIN_PID}"
echo "LOG=${LOG_PATH}"
