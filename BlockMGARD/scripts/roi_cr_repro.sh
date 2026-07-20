#!/usr/bin/env bash
#
# ROI compression-ratio comparison — reproduction driver
# ------------------------------------------------------
# Compares the compression ratio (CR) that BlockMGARD reaches with a ROI
# tolerance map against three uniform-error-bound baselines, on one variable
# per dataset:
#
#   blockmgard_roi  BlockMGARD with a ROI tolerance map (-roi -hh, ll 2 / gl 3)
#   mgard           plain MGARD-X            at the 1e-4 relative error level
#   cuzfp           cuZFP (fixed bitrate)    at the 1e-4 relative error level
#   cuszp           cuSZp (absolute bound)   at the 1e-4 relative error level
#
# The baselines use their pre-tuned 1e-4 parameters, which is the level closest
# to the ROI region's tolerance, so the comparison is "same ROI-region quality,
# whose CR is higher".
#
# Only the compression step is run: mgard-x reports both the compression ratio
# and (for the ROI run) the block-wise ROI verification during `-z`.
#
# Usage:
#   ./roi_cr_repro.sh                          # all datasets, all methods
#   ./roi_cr_repro.sh NYX Miranda              # only these datasets
#   METHODS="blockmgard_roi cuzfp" ./roi_cr_repro.sh   # only these methods
#   DRY_RUN=1 ./roi_cr_repro.sh                # print commands without running
#
set -euo pipefail

# ── Executables ───────────────────────────────────────────────────────────
# The three baselines live where install_baselines.sh puts them (same WORK_ROOT
# layout). BlockMGARD is the separate ROI-capable dev build, so the two MGARD
# installs must not share an LD_LIBRARY_PATH — each command sets its own.
WORK_ROOT="${WORK_ROOT:-/home/leonli/ROITest/comp}"
install_dir="${WORK_ROOT}/install"

MGARD_INSTALL="${WORK_ROOT}/mgard-x/install-cuda-hopper"
MGARD_EXEC="${MGARD_INSTALL}/bin/mgard-x"
ZFP_EXEC="${install_dir}/zfp/bin/zfp"
CUSZP_EXEC="${install_dir}/cuszp/bin/cuSZp"

BLOCKMGARD_INSTALL="${BLOCKMGARD_INSTALL:-/home/leonli/MGARD/install-cuda-hopper}"
BLOCKMGARD_EXEC="${BLOCKMGARD_INSTALL}/bin/mgard-x"

SDR_ROOT=/home/leonli/SDRBENCH

# ── Output locations ──────────────────────────────────────────────────────
RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/roi_cr_results.csv}"
RUN_LOG="${RUN_LOG:-$RESULTS_DIR/roi_cr_run.log}"
WORK_DIR="${WORK_DIR:-/home/leonli/ROITest/roi_cr_work}"   # ROI maps + compressed streams

# ROIGenerator is built on demand (single file, no dependencies).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIGEN_SRC="${SCRIPT_DIR}/../roi_generator/ROIGenerator.cpp"
ROIGEN_BIN="${SCRIPT_DIR}/../roi_generator/generate_roi_map"

# ── Fixed settings ────────────────────────────────────────────────────────
BASELINE_LEVEL="1e-4"          # error level the three baselines run at
ROI_LL=2                       # BlockMGARD local / global refactoring levels
ROI_GL=3

# ── Per-dataset configuration ─────────────────────────────────────────────
# One variable per dataset (the ones used in the ROI CR experiment).
# Dimension order follows each tool's own convention: the MGARD family takes the
# slowest dimension first, cuZFP/cuSZp take the reverse.
declare -A DS_FILE DS_MGARD_DT DS_ZFP_DT DS_CUSZP_DT
declare -A DS_MGARD_DIMS DS_ZFP_DIMS
declare -A DS_ROI_BG DS_ROI_SPEC
declare -A DS_MGARD_EB DS_ZFP_RATE DS_CUSZP_EB

# --- NYX / velocity_z ---
DS_FILE[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512/velocity_z.f32"
DS_MGARD_DT[NYX]="s"; DS_ZFP_DT[NYX]="-f"; DS_CUSZP_DT[NYX]="f32"
DS_MGARD_DIMS[NYX]="512 512 512"; DS_ZFP_DIMS[NYX]="512 512 512"
DS_ROI_BG[NYX]="1.3e-1"; DS_ROI_SPEC[NYX]="7e-3 256 280 256 280 256 280"
DS_MGARD_EB[NYX]="5.51e-3"; DS_ZFP_RATE[NYX]="14"; DS_CUSZP_EB[NYX]="3890"

# --- Hurricane / Wf48 ---
DS_FILE[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500/Wf48.bin.f32"
DS_MGARD_DT[Hurricane]="s"; DS_ZFP_DT[Hurricane]="-f"; DS_CUSZP_DT[Hurricane]="f32"
DS_MGARD_DIMS[Hurricane]="100 500 500"; DS_ZFP_DIMS[Hurricane]="500 500 100"
DS_ROI_BG[Hurricane]="1.3e-1"; DS_ROI_SPEC[Hurricane]="7e-3 16 40 256 280 256 280"
DS_MGARD_EB[Hurricane]="3.55e-3"; DS_ZFP_RATE[Hurricane]="14.7"; DS_CUSZP_EB[Hurricane]="0.0013333214"

# --- SCALE / V ---
DS_FILE[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200/V-98x1200x1200.f32"
DS_MGARD_DT[SCALE]="s"; DS_ZFP_DT[SCALE]="-f"; DS_CUSZP_DT[SCALE]="f32"
DS_MGARD_DIMS[SCALE]="98 1200 1200"; DS_ZFP_DIMS[SCALE]="1200 1200 98"
DS_ROI_BG[SCALE]="1.3e-1"; DS_ROI_SPEC[SCALE]="6.6e-3 16 40 256 280 256 280"
DS_MGARD_EB[SCALE]="3.77e-3"; DS_ZFP_RATE[SCALE]="10.9"; DS_CUSZP_EB[SCALE]="0.00593"

# --- Miranda / velocityz ---
DS_FILE[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384/velocityz.d64"
DS_MGARD_DT[Miranda]="d"; DS_ZFP_DT[Miranda]="-d"; DS_CUSZP_DT[Miranda]="f64"
DS_MGARD_DIMS[Miranda]="256 384 384"; DS_ZFP_DIMS[Miranda]="384 384 256"
DS_ROI_BG[Miranda]="1.3e-1"; DS_ROI_SPEC[Miranda]="3e-2 16 40 256 280 256 280"
DS_MGARD_EB[Miranda]="5.38e-3"; DS_ZFP_RATE[Miranda]="8.5"; DS_CUSZP_EB[Miranda]="0.0008996110"

# --- S3D / O2 ---
DS_FILE[S3D]="double_precision/SDRBENCH-S3D/sliced/O2.d64"
DS_MGARD_DT[S3D]="d"; DS_ZFP_DT[S3D]="-d"; DS_CUSZP_DT[S3D]="f64"
DS_MGARD_DIMS[S3D]="500 500 500"; DS_ZFP_DIMS[S3D]="500 500 500"
DS_ROI_BG[S3D]="1.3e-1"; DS_ROI_SPEC[S3D]="3e-2 16 40 256 280 256 280"
DS_MGARD_EB[S3D]="4.99e-3"; DS_ZFP_RATE[S3D]="10.15"; DS_CUSZP_EB[S3D]="0.0000208805208"

DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)
METHOD_ORDER=(blockmgard_roi mgard cuzfp cuszp)

# ── Selections ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  SELECTED_DATASETS=("$@")
else
  SELECTED_DATASETS=("${DATASET_ORDER[@]}")
fi
if [[ -n "${METHODS:-}" ]]; then
  read -r -a SELECTED_METHODS <<< "$METHODS"
else
  SELECTED_METHODS=("${METHOD_ORDER[@]}")
fi

CR_ROWS=()      # dataset,variable,method,error_level,compression_ratio

# ── Helpers ───────────────────────────────────────────────────────────────
strip_color() { sed -E 's/\x1b\[[0-9;]*m//g'; }

build_roigen() {
  [[ -n "${DRY_RUN:-}" ]] && return 0
  if [[ ! -x "$ROIGEN_BIN" || "$ROIGEN_SRC" -nt "$ROIGEN_BIN" ]]; then
    echo ">>> building generate_roi_map"
    g++ -O2 -std=c++11 "$ROIGEN_SRC" -o "$ROIGEN_BIN"
  fi
}

# Run a command, tee its output to the log, and echo it for parsing.
# Returns non-zero (with the output still logged) if the command fails.
run_capture() {
  local out
  # Trace goes to stderr: this function is called inside $( ), so anything on
  # stdout would be swallowed into the captured output instead of shown.
  { printf '+ '; printf '%q ' "$@"; echo; } >&2
  if [[ -n "${DRY_RUN:-}" ]]; then
    return 0
  fi
  if ! out="$("$@" 2>&1 | strip_color)"; then
    printf '%s\n' "$out" >>"$RUN_LOG"
    return 1
  fi
  printf '%s\n' "$out" >>"$RUN_LOG"
  printf '%s' "$out"
}

record() {   # dataset variable method level cr
  echo "    -> ${3} CR = ${5:-NA}"
  CR_ROWS+=("$1,$2,$3,$4,${5:-NA}")
}

# ── Per-method runners ────────────────────────────────────────────────────
run_blockmgard_roi() {
  local ds=$1 var=$2 in_data=$3
  local map="${WORK_DIR}/${ds}_roi.bin"
  local comp="${WORK_DIR}/${ds}_blockmgard_roi.mgard"
  local out cr violated

  echo "=== [blockmgard_roi] ${ds} / ${var} ==="
  run_capture "$ROIGEN_BIN" -o "$map" -dim ${DS_MGARD_DIMS[$ds]} \
      -bg "${DS_ROI_BG[$ds]}" -roi ${DS_ROI_SPEC[$ds]} >/dev/null || {
    echo "  !! ROI map generation failed" >&2; record "$ds" "$var" blockmgard_roi roi ""; return 0; }

  # -hh is required: the ROI tolerance map is only honoured by the hybrid
  # (block-local) hierarchy; without it the map is silently ignored.
  out=$(LD_LIBRARY_PATH="${BLOCKMGARD_INSTALL}/lib64:${BLOCKMGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}" \
        run_capture "$BLOCKMGARD_EXEC" -z -i "$in_data" -o "$comp" \
          -dt "${DS_MGARD_DT[$ds]}" -dim 3 ${DS_MGARD_DIMS[$ds]} \
          -em rel -r "$map" -roi -hh \
          -s inf -l huffman -d cuda -ll "$ROI_LL" -gl "$ROI_GL" -v 2) || {
    echo "  !! compression failed — see $RUN_LOG" >&2; record "$ds" "$var" blockmgard_roi roi ""; return 0; }

  [[ -n "${DRY_RUN:-}" ]] && return 0
  cr=$(awk '/\[info\] Compression ratio:/{print $NF; exit}' <<<"$out")
  # Surface the ROI verification so a silently-ignored map is obvious.
  violated=$(awk '/\[info\] Blocks violated:/{print $4, $5; exit}' <<<"$out")
  [[ -n "$violated" ]] && echo "    (ROI blocks violated: ${violated})"
  record "$ds" "$var" blockmgard_roi roi "$cr"
}

run_mgard() {
  local ds=$1 var=$2 in_data=$3
  local comp="${WORK_DIR}/${ds}_mgard.mgard" out cr

  echo "=== [mgard] ${ds} / ${var} @ ${BASELINE_LEVEL} ==="
  out=$(LD_LIBRARY_PATH="${MGARD_INSTALL}/lib:${MGARD_INSTALL}/lib64:${LD_LIBRARY_PATH:-}" \
        run_capture "$MGARD_EXEC" -z -i "$in_data" -o "$comp" \
          -dt "${DS_MGARD_DT[$ds]}" -dim 3 ${DS_MGARD_DIMS[$ds]} \
          -em rel -e "${DS_MGARD_EB[$ds]}" \
          -s inf -l huffman -d cuda -v 2) || {
    echo "  !! compression failed — see $RUN_LOG" >&2; record "$ds" "$var" mgard "$BASELINE_LEVEL" ""; return 0; }

  [[ -n "${DRY_RUN:-}" ]] && return 0
  cr=$(awk '/\[info\] Compression ratio:/{print $NF; exit}' <<<"$out")
  record "$ds" "$var" mgard "$BASELINE_LEVEL" "$cr"
}

run_cuzfp() {
  local ds=$1 var=$2 in_data=$3
  local comp="${WORK_DIR}/${ds}_zfp.zfp" out cr

  echo "=== [cuzfp] ${ds} / ${var} @ ${BASELINE_LEVEL} ==="
  out=$(run_capture "$ZFP_EXEC" -x cuda -i "$in_data" -z "$comp" \
          "${DS_ZFP_DT[$ds]}" -3 ${DS_ZFP_DIMS[$ds]} \
          -r "${DS_ZFP_RATE[$ds]}" -s) || {
    echo "  !! compression failed — see $RUN_LOG" >&2; record "$ds" "$var" cuzfp "$BASELINE_LEVEL" ""; return 0; }

  [[ -n "${DRY_RUN:-}" ]] && return 0
  cr=$(grep -oE 'ratio=[0-9.]+' <<<"$out" | head -1 | cut -d= -f2)
  record "$ds" "$var" cuzfp "$BASELINE_LEVEL" "$cr"
}

run_cuszp() {
  local ds=$1 var=$2 in_data=$3
  local comp="${WORK_DIR}/${ds}_cuszp.comp" out cr

  echo "=== [cuszp] ${ds} / ${var} @ ${BASELINE_LEVEL} ==="
  out=$(run_capture "$CUSZP_EXEC" -i "$in_data" -t "${DS_CUSZP_DT[$ds]}" -m plain \
          -eb abs "${DS_CUSZP_EB[$ds]}" -d 3 ${DS_ZFP_DIMS[$ds]} -x "$comp") || {
    echo "  !! compression failed — see $RUN_LOG" >&2; record "$ds" "$var" cuszp "$BASELINE_LEVEL" ""; return 0; }

  [[ -n "${DRY_RUN:-}" ]] && return 0
  cr=$(awk '/compression ratio:/{print $NF; exit}' <<<"$out")
  record "$ds" "$var" cuszp "$BASELINE_LEVEL" "$cr"
}

# ── Main loop ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
[[ -z "${DRY_RUN:-}" ]] && : > "$RUN_LOG"
build_roigen

for ds in "${SELECTED_DATASETS[@]}"; do
  if [[ -z "${DS_FILE[$ds]:-}" ]]; then
    echo "unknown dataset: $ds (valid: ${DATASET_ORDER[*]})" >&2
    exit 1
  fi
  in_data="${SDR_ROOT}/${DS_FILE[$ds]}"
  var="$(basename "${DS_FILE[$ds]}")"

  if [[ -z "${DRY_RUN:-}" && ! -f "$in_data" ]]; then
    echo "  !! input not found, skipping ${ds}: $in_data" >&2
    continue
  fi

  for m in "${SELECTED_METHODS[@]}"; do
    case "$m" in
      blockmgard_roi) run_blockmgard_roi "$ds" "$var" "$in_data" ;;
      mgard)          run_mgard          "$ds" "$var" "$in_data" ;;
      cuzfp)          run_cuzfp          "$ds" "$var" "$in_data" ;;
      cuszp)          run_cuszp          "$ds" "$var" "$in_data" ;;
      *) echo "unknown method: $m (valid: ${METHOD_ORDER[*]})" >&2; exit 1 ;;
    esac
  done
done

# ── Flush results ─────────────────────────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === compression ratio (one variable per dataset) ==="
    echo "# blockmgard_roi uses a ROI tolerance map; the baselines use their"
    echo "# pre-tuned ${BASELINE_LEVEL} relative-error parameters."
    echo "dataset,variable,method,error_level,compression_ratio"
    ((${#CR_ROWS[@]})) && printf '%s\n' "${CR_ROWS[@]}"
  } > "$RESULTS_FILE"

  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Full tool output:   $RUN_LOG"
fi
