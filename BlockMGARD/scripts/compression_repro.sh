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

# ── Executables / paths ───────────────────────────────────────────────────
# Two different builds, deliberately:
#   M-X  -> the stock MGARD-X baseline (the timing-instrumented 1.6.0 fork that
#           install_baselines.sh builds), the same binary roi_cr_repro.sh uses
#           for its `mgard` baseline.
#   BM   -> our BlockMGARD build.
# Running M-X out of our own build would compare BlockMGARD against a plain
# path that our branch has itself modified, which is not the baseline the
# comparison is about.
BASELINE_INSTALL="${BASELINE_INSTALL:-/home/leonli/ROITest/comp/mgard-x/install-cuda-hopper}"
BLOCKMGARD_INSTALL="${BLOCKMGARD_INSTALL:-/home/leonli/MGARD/install-cuda-hopper}"

declare -A MODE_EXEC MODE_LDPATH
MODE_EXEC[mx]="${BASELINE_INSTALL}/bin/mgard-x"
MODE_EXEC[bm]="${BLOCKMGARD_INSTALL}/bin/mgard-x"
MODE_EXEC[bmf]="${BLOCKMGARD_INSTALL}/bin/mgard-x"
MODE_LDPATH[mx]="${BASELINE_INSTALL}/lib64:${BASELINE_INSTALL}/lib"
MODE_LDPATH[bm]="${BLOCKMGARD_INSTALL}/lib64:${BLOCKMGARD_INSTALL}/lib"
MODE_LDPATH[bmf]="${BLOCKMGARD_INSTALL}/lib64:${BLOCKMGARD_INSTALL}/lib"

SDR_ROOT=/home/leonli/SDRBENCH
OUT_DATA=/home/leonli/ROITest/compressed.mgard          # scratch (overwritten)

RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/compression_results.csv}"
RUN_LOG="${RUN_LOG:-$RESULTS_DIR/compression_run.log}"

# ── Modes ─────────────────────────────────────────────────────────────────
# mx  : plain MGARD-X, standard multi-dim hierarchy
# bm  : BlockMGARD, hybrid hierarchy, kernel fusion OFF
# bmf : BlockMGARD, hybrid hierarchy, kernel fusion ON (MGARD's default)
#
# bm and bmf run the same hierarchy at the same error bound and differ only in
# whether the local decompose/recompose kernels are fused with quantization.
# Fusion is a pure performance transformation -- both reach the same error and
# the same compression ratio (to within the ~1% the compressed stream varies by
# on a GPU anyway), so the CR columns for the two are a cross-check, and the
# throughput columns are where they part.
BM_FLAGS="-hh -ll 1 -gl 2"
FIXED_FLAGS="-s inf -l huffman -d cuda -v 2"

MODE_ORDER=(mx bm bmf)
declare -A MODE_FLAGS
MODE_FLAGS[mx]=""
MODE_FLAGS[bm]="$BM_FLAGS -nkf"
MODE_FLAGS[bmf]="$BM_FLAGS"

# All three modes are measured by the same thing -- the kernel stage:
# decompose + quantize + lossless on the way in, and its inverse on the way
# out. Only the spelling differs, because the two builds print differently:
# the baseline writes "<Name> time: <s>" with throughput on a separate line,
# ours writes "<Name>: <s> (<N> GB/s)".
#
# Deliberately not the "low-level" or "high-level" timers: those add the
# per-subdomain prefetch layer and, in the baseline, a Calculate norm that
# takes 10 ms against our build's 0.2 ms. Comparing across two builds is only
# meaningful on the stage both measure the same way.
#
# Neither kernel timer prints a throughput, so it is derived below as
# uncompressed_bytes / seconds -- the definition MGARD uses for the figures it
# does print.
declare -A COMP_TIMER DECOMP_TIMER
COMP_TIMER[mx]="Compression Kernel time"
COMP_TIMER[bm]="Compression Kernel"
COMP_TIMER[bmf]="Compression Kernel"
DECOMP_TIMER[mx]="Decompression Kernel time"
DECOMP_TIMER[bm]="Decompression Kernel"
DECOMP_TIMER[bmf]="Decompression Kernel"

# Bytes of uncompressed data, for the throughput conversion.
dtype_bytes() { case "$1" in s) echo 4;; d) echo 8;; *) echo 0;; esac; }

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

# ── Run one (mode, dataset, variable, level, [record]) ────────────────────
# record=0 runs the identical command as a warm-up: no log, no parse, no row.
run_one() {
  local mode=$1 ds=$2 var=$3 level=$4 eb=$5 record=${6:-1}
  local in="${SDR_ROOT}/${DS_DIR[$ds]}/${var}"
  local extra="${MODE_FLAGS[$mode]}"

  if (( record )); then
    echo "=== [${mode}] ${ds}/${var} @ ${level} (e=${eb}) ==="
  else
    echo "  warm-up [${mode}] ${ds}/${var} @ ${level}"
  fi
  local cmd=("${MODE_EXEC[$mode]}" -z -i "$in" -o "$OUT_DATA" -dt "${DS_DT[$ds]}" -dim 3 ${DS_DIMS[$ds]} \
             -em rel -e "$eb" $extra $FIXED_FLAGS)
  if [[ -n "${DRY_RUN:-}" ]]; then printf '  '; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  if [[ ! -f "$in" ]]; then
    echo "  !! missing: $in" >&2
    (( record )) && RAW_ROWS+=("$ds,$var,$mode,$level,NA,NA,NA")
    return 0
  fi

  local out
  out="$(LD_LIBRARY_PATH="${MODE_LDPATH[$mode]}:${LD_LIBRARY_PATH:-}" \
         "${cmd[@]}" 2>&1 | strip_color)" || true
  (( record )) || return 0
  printf '%s\n' "$out" >>"$RUN_LOG"

  # Seconds from this mode's timer line, matched as a literal prefix. The
  # value is the field before the " s"; the kernel timers print nothing after
  # it, the "Low-level compression" line prints a "(N GB/s)" we ignore.
  secs() { awk -v P="[time] $1:" 'index($0, P) == 1 {
             for (i = 1; i <= NF; i++) if ($i == "s") { print $(i-1); exit }
           }' <<<"$out"; }

  local comp_s decomp_s cr
  comp_s=$(secs "${COMP_TIMER[$mode]}")
  decomp_s=$(secs "${DECOMP_TIMER[$mode]}")
  cr=$(awk '/\[info\] Compression ratio:/{print $NF; exit}' <<<"$out")

  # Throughput = uncompressed bytes / seconds, matching MGARD's own convention.
  local bytes comp_gbs decomp_gbs
  bytes=$(awk -v b="$(dtype_bytes "${DS_DT[$ds]}")" -v d="${DS_DIMS[$ds]}" \
              'BEGIN{n=b; c=split(d,a," "); for(i=1;i<=c;i++) n*=a[i]; print n}')
  comp_gbs=$(awk -v t="${comp_s:-}"   -v b="$bytes" 'BEGIN{print (t=="")?"NA":sprintf("%.6f", b/t/1e9)}')
  decomp_gbs=$(awk -v t="${decomp_s:-}" -v b="$bytes" 'BEGIN{print (t=="")?"NA":sprintf("%.6f", b/t/1e9)}')

  echo "    -> comp=${comp_gbs} GB/s (${comp_s:-NA}s, ${COMP_TIMER[$mode]})" \
       " decomp=${decomp_gbs} GB/s (${decomp_s:-NA}s, ${DECOMP_TIMER[$mode]})  CR=${cr:-NA}"
  RAW_ROWS+=("$ds,$var,$mode,$level,${comp_s:-NA},${comp_gbs},${decomp_s:-NA},${decomp_gbs},${cr:-NA}")
}

# ── Main loop ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
[[ -z "${DRY_RUN:-}" ]] && : > "$RUN_LOG"

# Validate the selection before running anything, so a typo fails now rather
# than after the warm-up has already burned several minutes.
for ds in "${SELECTED[@]}"; do
  [[ -n "${DS_DIR[$ds]:-}" ]] || { echo "unknown dataset: $ds" >&2; exit 1; }
done
for level in "${LEVELS[@]}"; do
  level_index "$level" >/dev/null || { echo "bad level: $level" >&2; exit 1; }
done

# One pass over the whole selection. record=0 discards everything it measures.
sweep() {
  local record=$1 ds var level i
  for ds in "${SELECTED[@]}"; do
    if (( record )); then echo "########## ${ds} ##########"; fi
    for var in ${DS_VARS[$ds]}; do
      local -a mx_eb bm_eb
      read -r -a mx_eb <<< "${MX_EB[$ds:$var]}"
      read -r -a bm_eb <<< "${BM_EB[$ds:$var]}"
      for level in "${LEVELS[@]}"; do
        i=$(level_index "$level")
        run_one mx  "$ds" "$var" "$level" "${mx_eb[$i]}" "$record"
        run_one bm  "$ds" "$var" "$level" "${bm_eb[$i]}" "$record"
        run_one bmf "$ds" "$var" "$level" "${bm_eb[$i]}" "$record"
      done
    done
    if (( record )); then echo; fi
  done
}

# ── Warm-up pass (results discarded) ──────────────────────────────────────
# The first run of a configuration is intermittently inflated by one to two
# orders of magnitude and looks like an ordinary measurement, so it silently
# skews whichever average it lands in. Run the full selection once and throw it
# away, as scaling_repro.sbatch does. WARMUP=0 skips it, roughly halving the
# runtime at that risk.
if [[ -z "${DRY_RUN:-}" && "${WARMUP:-1}" != "0" ]]; then
  echo "########## Warm-up pass (results discarded) ##########"
  sweep 0
  echo "Warm-up complete."
  echo
fi

sweep 1

# ── Flush: raw + Table II layout (averaged over variables) ────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-variable (kernel seconds, throughput GB/s, CR) ==="
    echo "dataset,variable,mode,level,comp_s,comp_gbs,decomp_s,decomp_gbs,cr"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === Table II: averaged over the four variables (M-X vs BM vs BMF) ==="
    echo "dataset,level,comp_MX_gbs,comp_BM_gbs,comp_BMF_gbs,decomp_MX_gbs,decomp_BM_gbs,decomp_BMF_gbs,CR_MX,CR_BM,CR_BMF"
    if ((${#RAW_ROWS[@]})); then
      printf '%s\n' "${RAW_ROWS[@]}" | awk -F, '
        # Accumulate per (dataset,level,mode), each column judged on its own:
        # a run whose timer did not parse still contributes its CR, and vice
        # versa. NA never matches the number test, so it is excluded rather
        # than coerced to zero.
        function num(v){ return v ~ /^-?[0-9.]+([eE][-+]?[0-9]+)?$/ }
        {
          key=$1","$4                       # dataset,level
          if(!(key in seen)){seen[key]=1; order[++k]=key}
          m=$3
          if(num($6)){ comp[key,m]+=$6; nc[key,m]++ }
          if(num($8)){ dec[key,m]+=$8;  nd[key,m]++ }
          if(num($9)){ cr[key,m]+=$9;   nr[key,m]++ }
        }
        function avg(key,m,col){
          if(col=="c") return nc[key,m] ? sprintf("%.2f", comp[key,m]/nc[key,m]) : "NA"
          if(col=="d") return nd[key,m] ? sprintf("%.2f", dec[key,m]/nd[key,m]) : "NA"
          return nr[key,m] ? sprintf("%.2f", cr[key,m]/nr[key,m]) : "NA" }
        END{ for(i=1;i<=k;i++){ key=order[i]
          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", key,
                 avg(key,"mx","c"), avg(key,"bm","c"), avg(key,"bmf","c"),
                 avg(key,"mx","d"), avg(key,"bm","d"), avg(key,"bmf","d"),
                 avg(key,"mx","r"), avg(key,"bm","r"), avg(key,"bmf","r") } }'
    fi
  } > "$RESULTS_FILE"
  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Full MGARD output:  $RUN_LOG"
fi
