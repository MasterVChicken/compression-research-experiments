#!/usr/bin/env bash
#
# ROI visualization experiments — results driver
# ----------------------------------------------
# Produces the decompressed (.dec) outputs used for the ROI visualization
# figures, for all three methods (BlockMGARD, cuZFP, cuSZp), for both
# experiments:
#
#   same_quality : fix visual quality, compare compression ratio  (SCALE/PRES)
#   same_cr      : fix compression ratio (~50), compare visual quality (Miranda/density)
#
# Only BlockMGARD uses the ROI tolerance map (built by ROIGenerator); the two
# baselines use a uniform error bound tuned to match BlockMGARD's operating
# point. The compression parameters below are the validated ones — do not
# change them; this script only wires up map generation, I/O and outputs.
#
# Usage:
#   ./roi_experiments.sh                 # run both experiments, all methods
#   ./roi_experiments.sh same_cr         # only one experiment (same_quality | same_cr)
#   DRY_RUN=1 ./roi_experiments.sh       # print the commands without running
#   OUT_DIR=/path ./roi_experiments.sh   # where .dec / maps / compressed go
#
set -euo pipefail

# ── Executables ───────────────────────────────────────────────────────────
zfp_exec=/home/leonli/ROITest/comp/install/zfp/bin/zfp
cuszp_exec=/home/leonli/ROITest/comp/install/cuszp/bin/cuSZp
# BlockMGARD uses the dev MGARD install (separate from the baseline one).
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
mgard_exec=${MGARD_INSTALL}/bin/mgard-x
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib64:${MGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────
SDR_ROOT=/home/leonli/SDRBENCH
OUT_DIR="${OUT_DIR:-/home/leonli/ROITest/roi_results}"   # .dec + maps + compressed
COMPRESSED="${OUT_DIR}/compressed.mgard"                  # scratch (reused per run)

# BlockMGARD error mode for the tolerance map: rel (default) or abs.
# A/B test with:  EM=abs ./roi_experiments.sh
EM="${EM:-rel}"

# ROIGenerator lives beside this script's repo; build-if-missing (zero deps).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIGEN_SRC="${SCRIPT_DIR}/../roi_generator/ROIGenerator.cpp"
ROIGEN_BIN="${SCRIPT_DIR}/../roi_generator/generate_roi_map"

# Executable output goes to this log instead of the terminal (see run()).
RUN_LOG="${OUT_DIR}/roi_run.log"

# Strip ANSI colour codes from tool output (mgard colours its log lines).
strip_color() { sed -E 's/\x1b\[[0-9;]*m//g'; }

# ── Helpers ───────────────────────────────────────────────────────────────
run() {   # echo the command, then run it (unless DRY_RUN)
  printf '+ '; printf '%q ' "$@"; echo
  [[ -n "${DRY_RUN:-}" ]] && return 0
  if [[ -n "${VERBOSE:-}" ]]; then
    "$@" 2>&1 | strip_color                     # VERBOSE=1 -> decoloured output, live
  elif ! { "$@" 2>&1 | strip_color >>"$RUN_LOG"; }; then
    echo "  !! command failed — see $RUN_LOG" >&2
    return 1                   # propagate failure (set -e will stop the script)
  fi
}

build_roigen() {
  [[ -n "${DRY_RUN:-}" ]] && return 0
  if [[ ! -x "$ROIGEN_BIN" || "$ROIGEN_SRC" -nt "$ROIGEN_BIN" ]]; then
    echo ">>> building generate_roi_map"
    g++ -O2 -std=c++11 "$ROIGEN_SRC" -o "$ROIGEN_BIN"
  fi
}

# BlockMGARD with ROI map:  generate map -> compress -> decompress to .dec
#   $1 in  $2 dt(s|d)  $3 mgard_dims  $4 ll  $5 gl  $6 out_dec
#   $7 roigen_dims  $8 bg  $9 roi_spec("tol x0 x1 y0 y1 z0 z1")  ${10} roi_map
blockmgard() {
  local in=$1 dt=$2 mdims=$3 ll=$4 gl=$5 out=$6 rgdims=$7 bg=$8 roi=$9 map=${10}
  echo "=== BlockMGARD -> $(basename "$out") ==="
  run "$ROIGEN_BIN" -o "$map" -dim $rgdims -bg $bg -roi $roi
  # -hh enables the hybrid (block-local) hierarchy — the ONLY path that
  # consumes the ROI tolerance map. Without it the map is silently ignored.
  run "$mgard_exec" -z -i "$in" -o "$COMPRESSED" -dt $dt -dim 3 $mdims \
      -em $EM -r "$map" -roi -hh -s inf -l huffman -d cuda -ll $ll -gl $gl -v 2
  run "$mgard_exec" -x -i "$COMPRESSED" -o "$out" -r "$map" -roi \
      -ll $ll -gl $gl -em $EM -orig "$in" -d cuda -v 1
}

# cuZFP baseline: compress (with stats) -> decompress to .dec
#   $1 in  $2 dtflag(-f|-d)  $3 zfp_dims  $4 rate  $5 out_dec
zfp_baseline() {
  local in=$1 dtf=$2 zdims=$3 rate=$4 out=$5 comp="${5%.dec}.zfp"
  echo "=== cuZFP -> $(basename "$out") ==="
  run "$zfp_exec" -x cuda -i "$in" -z "$comp" $dtf -3 $zdims -r $rate -s
  run "$zfp_exec" -x cuda -z "$comp" -o "$out" $dtf -3 $zdims -r $rate
}

# cuSZp baseline: compress + decompress to .dec in one call
#   $1 in  $2 type(f32|f64)  $3 cuszp_dims  $4 eb_abs  $5 out_dec
cuszp_baseline() {
  local in=$1 t=$2 cdims=$3 eb=$4 out=$5 comp="${5%.dec}.comp"
  echo "=== cuSZp -> $(basename "$out") ==="
  run "$cuszp_exec" -i "$in" -t $t -m plain -eb abs $eb -d 3 $cdims \
      -x "$comp" -o "$out"
}

# ── Experiment 1: same visual quality (SCALE / PRES, f32) ─────────────────
exp_same_quality() {
  local in="${SDR_ROOT}/single_precision/SDRBENCH-SCALE_98x1200x1200/PRES-98x1200x1200.f32"
  echo "########## Experiment: same_quality (SCALE / PRES) ##########"
  blockmgard "$in" s "98 1200 1200" 2 2 "${OUT_DIR}/blockmgard_scale_pres.dec" \
             "98 1200 1200" 7e-1 "6e-5 72 80 900 1050 0 150" "${OUT_DIR}/scale_pres_roi.bin"
  zfp_baseline   "$in" -f "1200 1200 98" 9.2            "${OUT_DIR}/zfp_scale_pres.dec"
  cuszp_baseline "$in" f32 "1200 1200 98" 0.101820218750 "${OUT_DIR}/cuszp_scale_pres.dec"
}

# ── Experiment 2: same CR ~50 (Miranda / density, f64) ────────────────────
exp_same_cr() {
  local in="${SDR_ROOT}/double_precision/SDRBENCH-Miranda-256x384x384/density.d64"
  echo "########## Experiment: same_cr (Miranda / density) ##########"
  blockmgard "$in" d "256 384 384" 2 3 "${OUT_DIR}/blockmgard_miranda_density.dec" \
             "256 384 384" 1.1e+0 "7e-3 120 128 70 120 40 90" "${OUT_DIR}/miranda_density_roi.bin"
  zfp_baseline   "$in" -d "384 384 256" 1.28 "${OUT_DIR}/zfp_miranda_density.dec"
  cuszp_baseline "$in" f64 "384 384 256" 2    "${OUT_DIR}/cuszp_miranda_density.dec"
}

# ── Select experiments and run ────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  EXPS=("$@")
else
  EXPS=(same_quality same_cr)
fi

mkdir -p "$OUT_DIR"
[[ -z "${DRY_RUN:-}" && -z "${VERBOSE:-}" ]] && : > "$RUN_LOG"   # fresh log per run
build_roigen

for e in "${EXPS[@]}"; do
  case "$e" in
    same_quality) exp_same_quality ;;
    same_cr)      exp_same_cr ;;
    *) echo "unknown experiment: $e (valid: same_quality same_cr)" >&2; exit 1 ;;
  esac
done

echo
echo "Done. Decompressed outputs (.dec) are in: $OUT_DIR"
[[ -z "${VERBOSE:-}" ]] && echo "Executable output was logged to: $RUN_LOG"
