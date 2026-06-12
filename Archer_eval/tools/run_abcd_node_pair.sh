#!/usr/bin/env bash
#
# Launch two logical workers on one 8-GPU node in 4+4 mode.
# Example:
#   bash tools/run_abcd_node_pair.sh B1 B2
#

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <WORKER_A> <WORKER_B>   (e.g. B1 B2)" >&2
  exit 2
fi

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"
WORKER_SCRIPT="${EVAL_DIR}/tools/eval_abcd_worker.sh"

W1="$1"
W2="$2"

DATASET="${DATASET:-livecodebench_v5}"
STEPS_CSV="${STEPS_CSV:-10,50,80,105}"
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}"
N_GPUS="${N_GPUS:-4}"

mkdir -p "${EVAL_DIR}/tools/logs"
TS="$(date +%Y%m%d_%H%M%S)"

decode_worker() {
  local w="$1"
  GROUP="${w:0:1}"
  INDEX="${w:1}"
  case "${GROUP}" in
    A) TOTAL=2 ;;
    B) TOTAL=4 ;;
    C) TOTAL=4 ;;
    D) TOTAL=2 ;;
    E) TOTAL=2 ;;
    F) TOTAL=2 ;;
    *)
      echo "Invalid worker ${w}. Must look like A1/B3/C2/D1." >&2
      return 1
      ;;
  esac
  if ! [[ "${INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Invalid worker index in ${w}" >&2
    return 1
  fi
  if (( INDEX < 1 || INDEX > TOTAL )); then
    echo "Worker ${w} out of range for group ${GROUP} (1..${TOTAL})" >&2
    return 1
  fi
}

run_one() {
  local worker="$1"
  local cuda="$2"
  local log_file="$3"
  decode_worker "${worker}"
  CUDA_VISIBLE_DEVICES="${cuda}" \
    GROUP="${GROUP}" \
    WORKER_INDEX="${INDEX}" \
    WORKER_TOTAL="${TOTAL}" \
    DATASET="${DATASET}" \
    STEPS_CSV="${STEPS_CSV}" \
    N_GPUS="${N_GPUS}" \
    MERGE_IF_NEEDED="${MERGE_IF_NEEDED}" \
    bash "${WORKER_SCRIPT}" > "${log_file}" 2>&1 &
  RUN_PID=$!
}

LOG1="${EVAL_DIR}/tools/logs/nodepair_${W1}_${DATASET}_${TS}.log"
LOG2="${EVAL_DIR}/tools/logs/nodepair_${W2}_${DATASET}_${TS}.log"

run_one "${W1}" "0,1,2,3" "${LOG1}"
PID1="${RUN_PID}"
run_one "${W2}" "4,5,6,7" "${LOG2}"
PID2="${RUN_PID}"

echo "Started ${W1} pid=${PID1}, log=${LOG1}"
echo "Started ${W2} pid=${PID2}, log=${LOG2}"

wait "${PID1}"
S1=$?
wait "${PID2}"
S2=$?

echo "Exit codes: ${W1}=${S1}, ${W2}=${S2}"
if [[ ${S1} -ne 0 || ${S2} -ne 0 ]]; then
  exit 1
fi
