#!/usr/bin/env bash
#
# MGARD-X vs BlockMGARD throughput & CR — reproduction driver (paper Table II)
# ---------------------------------------------------------------------------
# For every dataset and error level, runs plain MGARD-X (M-X) and BlockMGARD
# (BM, hybrid `-hh -ll 1 -gl 2`) and reports compression throughput,
# decompression throughput and compression ratio, AVERAGED over the four
# variables of each dataset.
#
# Both use their own tuned relative error bounds — M-X from mgard_comp.sh, BM
# from block_mgard_comp.sh — chosen so the ACHIEVED error is 1e-2 / 1e-4 / 1e-6
# (MGARD's `-e` is a relative bound in its own norm, so the input values look
# larger than the achieved error).
#
# Throughput is the kernel-level "Compress/Decompress pipeline" figure MGARD
# prints (decompose + quantize + lossless, excluding host<->device transfer).
#
# Usage:
#   ./compression_repro.sh                    # all datasets, all levels
#   ./compression_repro.sh NYX Miranda        # only these datasets
#   LEVEL=1e-4 ./compression_repro.sh         # only one level (1e-2|1e-4|1e-6)
#   DRY_RUN=1 ./compression_repro.sh          # print commands without running
#
set -uo pipefail

# ── Executable / paths ────────────────────────────────────────────────────
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
EXEC=${MGARD_INSTALL}/bin/mgard-x
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib64:${MGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

SDR_ROOT=/home/leonli/SDRBENCH
OUT_DATA=/home/leonli/ROITest/compressed.mgard          # scratch (overwritten)

RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/compression_results.csv}"
RUN_LOG="${RUN_LOG:-$RESULTS_DIR/compression_run.log}"

# ── Fixed BlockMGARD hierarchy config (M-X uses neither) ───────────────────
BM_FLAGS="-hh -ll 1 -gl 2"
FIXED_FLAGS="-s inf -l huffman -d cuda -v 2"

# ── Per-dataset config: dir | dtype | dims | variables ────────────────────
declare -A DS_DIR DS_DT DS_DIMS DS_VARS
DS_DIR[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DT[NYX]="s"; DS_DIMS[NYX]="512 512 512"
DS_VARS[NYX]="temperature.f32 velocity_x.f32 velocity_y.f32 velocity_z.f32"

DS_DIR[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500"
DS_DT[Hurricane]="s"; DS_DIMS[Hurricane]="100 500 500"
DS_VARS[Hurricane]="Pf48.bin.f32 Uf48.bin.f32 Vf48.bin.f32 Wf48.bin.f32"

DS_DIR[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200"
DS_DT[SCALE]="s"; DS_DIMS[SCALE]="98 1200 1200"
DS_VARS[SCALE]="PRES-98x1200x1200.f32 T-98x1200x1200.f32 U-98x1200x1200.f32 V-98x1200x1200.f32"

DS_DIR[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384"
DS_DT[Miranda]="d"; DS_DIMS[Miranda]="256 384 384"
DS_VARS[Miranda]="density.d64 diffusivity.d64 pressure.d64 velocityz.d64"

DS_DIR[S3D]="double_precision/SDRBENCH-S3D/sliced"
DS_DT[S3D]="d"; DS_DIMS[S3D]="500 500 500"
DS_VARS[S3D]="CH4.d64 CO2.d64 H2O.d64 O2.d64"

DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)
ALL_LEVELS=(1e-2 1e-4 1e-6)

# ── Tuned error bounds:  "ds:var" -> "1e-2 1e-4 1e-6" ─────────────────────
# M-X from mgard_comp.sh, BM from block_mgard_comp.sh.
declare -A MX_EB BM_EB
MX_EB[NYX:temperature.f32]="6.15e-1 5.34e-3 4.99e-5"
MX_EB[NYX:velocity_x.f32]="5.9e-1 5.43e-3 5.45e-5"
MX_EB[NYX:velocity_y.f32]="5.68e-1 5.54e-3 5.5e-5"
MX_EB[NYX:velocity_z.f32]="5.52e-1 5.51e-3 5.15e-5"
MX_EB[Hurricane:Pf48.bin.f32]="4.71e-1 3.78e-3 3.96e-5"
MX_EB[Hurricane:Uf48.bin.f32]="3.84e-1 3.82e-3 3.82e-5"
MX_EB[Hurricane:Vf48.bin.f32]="3.49e-1 3.82e-3 3.8e-5"
MX_EB[Hurricane:Wf48.bin.f32]="3.27e-1 3.55e-3 3e-5"
MX_EB[SCALE:PRES-98x1200x1200.f32]="4.86e-1 3.99e-3 3.72e-5"
MX_EB[SCALE:T-98x1200x1200.f32]="3.89e-1 3.85e-3 3.52e-5"
MX_EB[SCALE:U-98x1200x1200.f32]="4.1e-1 3.87e-3 5e-5"
MX_EB[SCALE:V-98x1200x1200.f32]="3.9e-1 3.77e-3 5e-5"
MX_EB[Miranda:density.d64]="5.38e-1 5.38e-3 4.98e-5"
MX_EB[Miranda:diffusivity.d64]="5.38e-1 5.38e-3 5.2e-5"
MX_EB[Miranda:pressure.d64]="5.38e-1 5.38e-3 4.98e-5"
MX_EB[Miranda:velocityz.d64]="5.58e-1 5.38e-3 4.8e-5"
MX_EB[S3D:CH4.d64]="4.9e-1 4.99e-3 4.79e-5"
MX_EB[S3D:CO2.d64]="5.5e-1 5.45e-3 4.9e-5"
MX_EB[S3D:H2O.d64]="4.9e-1 4.92e-3 4.79e-5"
MX_EB[S3D:O2.d64]="4.9e-1 4.99e-3 4.99e-5"

BM_EB[NYX:temperature.f32]="4.8e-1 3.9e-3 3.83e-5"
BM_EB[NYX:velocity_x.f32]="4.2e-1 3.83e-3 3.74e-5"
BM_EB[NYX:velocity_y.f32]="4.45e-1 3.6e-3 3.81e-5"
BM_EB[NYX:velocity_z.f32]="4.23e-1 3.86e-3 3.81e-5"
BM_EB[Hurricane:Pf48.bin.f32]="4.36e-1 3.97e-3 4.53e-5"
BM_EB[Hurricane:Uf48.bin.f32]="4.25e-1 4e-3 4.53e-5"
BM_EB[Hurricane:Vf48.bin.f32]="4.36e-1 3.99e-3 4.53e-5"
BM_EB[Hurricane:Wf48.bin.f32]="4.24e-1 3.89e-3 4.53e-5"
BM_EB[SCALE:PRES-98x1200x1200.f32]="6.53e-1 3.97e-3 3.6e-5"
BM_EB[SCALE:T-98x1200x1200.f32]="4.1e-1 3.82e-3 3.2e-5"
BM_EB[SCALE:U-98x1200x1200.f32]="3.84e-1 3.74e-3 3.6e-5"
BM_EB[SCALE:V-98x1200x1200.f32]="4.1e-1 3.69e-3 3.6e-5"
BM_EB[Miranda:density.d64]="5.06e-1 4.11e-3 4.09e-5"
BM_EB[Miranda:diffusivity.d64]="5.04e-1 4.1e-3 4.08e-5"
BM_EB[Miranda:pressure.d64]="5.05e-1 4.11e-3 4.08e-5"
BM_EB[Miranda:velocityz.d64]="5.05e-1 4.11e-3 4.095e-5"
BM_EB[S3D:CH4.d64]="4.899e-1 4.005e-3 3.86e-5"
BM_EB[S3D:CO2.d64]="4.838e-1 4.103e-3 3.874e-5"
BM_EB[S3D:H2O.d64]="4.82e-1 4e-3 3.848e-5"
BM_EB[S3D:O2.d64]="4.988e-1 4.101e-3 3.908e-5"

# ── Selections ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then SELECTED=("$@"); else SELECTED=("${DATASET_ORDER[@]}"); fi
if [[ -n "${LEVEL:-}" ]]; then IFS=',' read -r -a LEVELS <<< "$LEVEL"; else LEVELS=("${ALL_LEVELS[@]}"); fi
level_index() { case "$1" in 1e-2) echo 0;; 1e-4) echo 1;; 1e-6) echo 2;; *) return 1;; esac; }

strip_color() { sed -E 's/\x1b\[[0-9;]*m//g'; }
RAW_ROWS=()   # dataset,variable,mode,level,comp_gbs,decomp_gbs,cr

# ── Run one (mode, dataset, variable, level) ──────────────────────────────
run_one() {
  local mode=$1 ds=$2 var=$3 level=$4 eb=$5
  local in="${SDR_ROOT}/${DS_DIR[$ds]}/${var}"
  local extra=""; [[ "$mode" == "bm" ]] && extra="$BM_FLAGS"

  echo "=== [${mode}] ${ds}/${var} @ ${level} (e=${eb}) ==="
  local cmd=("$EXEC" -z -i "$in" -o "$OUT_DATA" -dt "${DS_DT[$ds]}" -dim 3 ${DS_DIMS[$ds]} \
             -em rel -e "$eb" $extra $FIXED_FLAGS)
  if [[ -n "${DRY_RUN:-}" ]]; then printf '  '; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  [[ -f "$in" ]] || { echo "  !! missing: $in" >&2; RAW_ROWS+=("$ds,$var,$mode,$level,NA,NA,NA"); return 0; }

  local out
  out="$("${cmd[@]}" 2>&1 | strip_color)" || true
  printf '%s\n' "$out" >>"$RUN_LOG"

  # Parse the GB/s inside "(...)" on the pipeline lines, and the CR.
  local comp decomp cr
  comp=$(awk '/\[time\] Compress pipeline:/  {for(i=1;i<=NF;i++) if($i ~ /GB\/s/){v=$(i-1); gsub(/[()]/,"",v); print v; exit}}' <<<"$out")
  decomp=$(awk '/\[time\] Decompress pipeline:/{for(i=1;i<=NF;i++) if($i ~ /GB\/s/){v=$(i-1); gsub(/[()]/,"",v); print v; exit}}' <<<"$out")
  cr=$(awk '/\[info\] Compression ratio:/{print $NF; exit}' <<<"$out")

  echo "    -> comp=${comp:-NA} GB/s  decomp=${decomp:-NA} GB/s  CR=${cr:-NA}"
  RAW_ROWS+=("$ds,$var,$mode,$level,${comp:-NA},${decomp:-NA},${cr:-NA}")
}

# ── Main loop ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
[[ -z "${DRY_RUN:-}" ]] && : > "$RUN_LOG"

for ds in "${SELECTED[@]}"; do
  [[ -n "${DS_DIR[$ds]:-}" ]] || { echo "unknown dataset: $ds" >&2; exit 1; }
  echo "########## ${ds} ##########"
  for var in ${DS_VARS[$ds]}; do
    read -r -a mx <<< "${MX_EB[$ds:$var]}"
    read -r -a bm <<< "${BM_EB[$ds:$var]}"
    for level in "${LEVELS[@]}"; do
      i=$(level_index "$level") || { echo "bad level: $level" >&2; exit 1; }
      run_one mx "$ds" "$var" "$level" "${mx[$i]}"
      run_one bm "$ds" "$var" "$level" "${bm[$i]}"
    done
  done
  echo
done

# ── Flush: raw + Table II layout (averaged over variables) ────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-variable (throughput GB/s, CR) ==="
    echo "dataset,variable,mode,level,comp_gbs,decomp_gbs,cr"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === Table II: averaged over the four variables (M-X vs BM) ==="
    echo "dataset,level,comp_MX_gbs,comp_BM_gbs,decomp_MX_gbs,decomp_BM_gbs,CR_MX,CR_BM"
    if ((${#RAW_ROWS[@]})); then
      printf '%s\n' "${RAW_ROWS[@]}" | awk -F, '
        $5 ~ /^[0-9.eE+-]+$/ {
          key=$1","$4                       # dataset,level
          if(!(key in seen)){seen[key]=1; order[++k]=key}
          if($3=="mx"){ nc_mx[key]++; comp_mx[key]+=$5; dec_mx[key]+=$6; cr_mx[key]+=$7 }
          else        { nc_bm[key]++; comp_bm[key]+=$5; dec_bm[key]+=$6; cr_bm[key]+=$7 }
        }
        function avg(s,n){ return n>0 ? s/n : 0 }
        END{ for(i=1;i<=k;i++){ key=order[i]
          printf "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", key,
                 avg(comp_mx[key],nc_mx[key]), avg(comp_bm[key],nc_bm[key]),
                 avg(dec_mx[key],nc_mx[key]),  avg(dec_bm[key],nc_bm[key]),
                 avg(cr_mx[key],nc_mx[key]),   avg(cr_bm[key],nc_bm[key]) } }'
    fi
  } > "$RESULTS_FILE"
  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Full MGARD output:  $RUN_LOG"
fi
