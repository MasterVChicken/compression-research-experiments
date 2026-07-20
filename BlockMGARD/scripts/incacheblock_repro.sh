#!/usr/bin/env bash
#
# In-cache-block ablation — reproduction driver
# ---------------------------------------------
# Compares two hierarchy configurations of BlockMGARD at a fixed 1e-2 relative
# error bound, over all SDRBENCH datasets:
#
#   local   Local Quantization Only  (-ll 1 -gl 0)
#   global  Global Quantization Only (-ll 0 -gl 1)
#
# For each run it extracts the decomposition / recomposition times matching the
# mode (Local* for the local runs, Global* for the global runs) and averages
# them over the four variables of each dataset.
#
# A single `-z` run reports both times: mgard-x decompresses right after
# compressing in order to print statistics.
#
# Usage:
#   ./incacheblock_repro.sh                   # both modes, all datasets
#   ./incacheblock_repro.sh NYX Miranda       # only these datasets
#   MODE=local ./incacheblock_repro.sh        # only one mode (local | global)
#   DRY_RUN=1 ./incacheblock_repro.sh         # print commands without running
#
set -euo pipefail

# ── Executable / paths ────────────────────────────────────────────────────
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
EXEC=${MGARD_INSTALL}/bin/mgard-x
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib64:${MGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

SDR_ROOT=/home/leonli/SDRBENCH
OUT_DATA=/home/leonli/ROITest/compressed.dat     # scratch (overwritten each run)

RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/incacheblock_results.csv}"
RUN_LOG="${RUN_LOG:-$RESULTS_DIR/incacheblock_run.log}"

# ── Fixed compression settings ────────────────────────────────────────────
ERR_MODE="rel"
ERR_BOUND="1e-2"
FIXED_FLAGS="-hh -s inf -l huffman -d cuda -v 2"

# ── Modes: name -> "<-ll> <-gl>" and the timer label prefix to extract ─────
declare -A MODE_LEVELS MODE_LABEL
MODE_LEVELS[local]="-ll 1 -gl 0";  MODE_LABEL[local]="Local"
MODE_LEVELS[global]="-ll 0 -gl 1"; MODE_LABEL[global]="Global"
MODE_ORDER=(local global)

# ── Per-dataset config: data dir (under SDR_ROOT), dtype, mgard dims ───────
declare -A DS_DIR DS_DTYPE DS_DIMS
DS_DIR[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DTYPE[NYX]="s";  DS_DIMS[NYX]="512 512 512"

DS_DIR[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500"
DS_DTYPE[Hurricane]="s";  DS_DIMS[Hurricane]="100 500 500"

DS_DIR[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200"
DS_DTYPE[SCALE]="s";  DS_DIMS[SCALE]="98 1200 1200"

DS_DIR[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384"
DS_DTYPE[Miranda]="d";  DS_DIMS[Miranda]="256 384 384"

DS_DIR[S3D]="double_precision/SDRBENCH-S3D/sliced"
DS_DTYPE[S3D]="d";  DS_DIMS[S3D]="500 500 500"

DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)

# ── Per-dataset variables (four each) ─────────────────────────────────────
declare -A DS_VARS
DS_VARS[NYX]="temperature.f32 velocity_x.f32 velocity_y.f32 velocity_z.f32"
DS_VARS[Hurricane]="Pf48.bin.f32 Uf48.bin.f32 Vf48.bin.f32 Wf48.bin.f32"
DS_VARS[SCALE]="PRES-98x1200x1200.f32 T-98x1200x1200.f32 U-98x1200x1200.f32 V-98x1200x1200.f32"
DS_VARS[Miranda]="density.d64 diffusivity.d64 pressure.d64 velocityz.d64"
DS_VARS[S3D]="CH4.d64 CO2.d64 H2O.d64 O2.d64"

# ── Selections ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  SELECTED_DATASETS=("$@")
else
  SELECTED_DATASETS=("${DATASET_ORDER[@]}")
fi

if [[ -n "${MODE:-}" ]]; then
  IFS=',' read -r -a SELECTED_MODES <<< "$MODE"
else
  SELECTED_MODES=("${MODE_ORDER[@]}")
fi

# Raw per-run rows: mode,dataset,variable,decomposition_s,recomposition_s
RAW_ROWS=()

# ── Helpers ───────────────────────────────────────────────────────────────
strip_color() { sed -E 's/\x1b\[[0-9;]*m//g'; }

run_one() {
  local mode=$1 ds=$2 var=$3
  local in_data="${SDR_ROOT}/${DS_DIR[$ds]}/${var}"
  local dt="${DS_DTYPE[$ds]}"
  local dims="${DS_DIMS[$ds]}"
  local levels="${MODE_LEVELS[$mode]}"
  local label="${MODE_LABEL[$mode]}"

  echo "=== [${mode}] ${ds} / ${var} ==="
  local cmd=("$EXEC" -z -i "$in_data" -o "$OUT_DATA" -dt "$dt" -dim 3 $dims \
             -em "$ERR_MODE" -e "$ERR_BOUND" $FIXED_FLAGS $levels)

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  if [[ ! -f "$in_data" ]]; then
    echo "  !! input not found, skipping: $in_data" >&2
    return 0
  fi

  local out
  out="$("${cmd[@]}" 2>&1 | strip_color)" || {
    echo "  !! command failed — see $RUN_LOG" >&2
    printf '%s\n' "$out" >>"$RUN_LOG"
    return 1
  }
  printf '%s\n' "$out" >>"$RUN_LOG"

  # "[time] Local Decomposition: 0.004355 s (129.610372 GB/s)" -> field 4
  local dec rec
  dec=$(awk -v L="$label" '$0 ~ ("\\[time\\] " L " Decomposition:") {print $4; exit}' <<<"$out")
  rec=$(awk -v L="$label" '$0 ~ ("\\[time\\] " L " Recomposition:") {print $4; exit}' <<<"$out")

  echo "    -> ${label} decomposition=${dec:-NA}s  recomposition=${rec:-NA}s"
  RAW_ROWS+=("$mode,$ds,$var,${dec:-NA},${rec:-NA}")
}

# ── Main loop ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
[[ -z "${DRY_RUN:-}" ]] && : > "$RUN_LOG"

for mode in "${SELECTED_MODES[@]}"; do
  if [[ -z "${MODE_LEVELS[$mode]:-}" ]]; then
    echo "unknown mode: $mode (valid: ${MODE_ORDER[*]})" >&2
    exit 1
  fi
  echo "########## Mode: ${mode} (${MODE_LEVELS[$mode]}) ##########"
  for ds in "${SELECTED_DATASETS[@]}"; do
    if [[ -z "${DS_DIR[$ds]:-}" ]]; then
      echo "unknown dataset: $ds (valid: ${DATASET_ORDER[*]})" >&2
      exit 1
    fi
    for var in ${DS_VARS[$ds]}; do
      run_one "$mode" "$ds" "$var"
    done
  done
done

# ── Flush: raw rows + per-dataset averages over the four variables ────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-variable timings (seconds) ==="
    echo "mode,dataset,variable,decomposition_s,recomposition_s"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === per-dataset averages over variables (seconds) ==="
    echo "mode,dataset,num_variables,avg_decomposition_s,avg_recomposition_s"
    if ((${#RAW_ROWS[@]})); then
      printf '%s\n' "${RAW_ROWS[@]}" | awk -F, '
        # Average only rows whose two timings parsed as numbers.
        $4 ~ /^[0-9.eE+-]+$/ && $5 ~ /^[0-9.eE+-]+$/ {
          key = $1 "," $2
          if (!(key in n)) order[++k] = key
          n[key]++; d[key] += $4; r[key] += $5
        }
        END {
          for (i = 1; i <= k; i++) {
            key = order[i]
            printf "%s,%d,%.6f,%.6f\n", key, n[key], d[key]/n[key], r[key]/n[key]
          }
        }'
    fi
  } > "$RESULTS_FILE"
  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Full tool output:   $RUN_LOG"
fi
