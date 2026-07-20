#!/usr/bin/env bash
#
# Hybrid hierarchy ablation — reproduction driver
# -----------------------------------------------
# Sweeps the local/global refactoring-level configurations of BlockMGARD over
# all five SDRBENCH datasets (four variables each) and reports the
# decomposition / recomposition times, averaged over the four variables.
#
# Configurations (-ll / -gl):
#   globalmax  0 / per-dataset max (NYX 9, Hurricane 7, SCALE 7, Miranda 8, S3D 9)
#   l1g2       1 / 2
#   l2g1       2 / 1
#   local5     5 / 0
#   local3     3 / 0
#   local1     1 / 0
#
# Local and global times are recorded separately; whichever side a configuration
# does not use (level 0) is recorded as 0.
#
# Each variable has its own tuned relative error bound (see the EB table below);
# a single `-z` run reports both decomposition and recomposition times because
# mgard-x decompresses right after compressing to print statistics.
#
# Usage:
#   ./hybridhierarchy_repro.sh                    # all configs, all datasets
#   ./hybridhierarchy_repro.sh NYX Miranda        # only these datasets
#   CONFIG=local1 ./hybridhierarchy_repro.sh      # only one config (comma-separated ok)
#   DRY_RUN=1 ./hybridhierarchy_repro.sh          # print commands without running
#
set -euo pipefail

# ── Executable / paths ────────────────────────────────────────────────────
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
EXEC=${MGARD_INSTALL}/bin/mgard-x
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib64:${MGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

SDR_ROOT=/home/leonli/SDRBENCH
OUT_DATA=/home/leonli/ROITest/compressed.dat     # scratch (overwritten each run)

RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/hybridhierarchy_results.csv}"
RUN_LOG="${RUN_LOG:-$RESULTS_DIR/hybridhierarchy_run.log}"

# ── Fixed compression settings ────────────────────────────────────────────
ERR_MODE="rel"
FIXED_FLAGS="-hh -s inf -l huffman -d cuda -v 2"

# ── Configurations: "-ll -gl", optionally overridden per dataset ──────────
# Key "<config>" is the default; key "<config>:<dataset>" overrides it.
declare -A CFG_LEVELS
CFG_LEVELS[globalmax:NYX]="0 9"
CFG_LEVELS[globalmax:Hurricane]="0 7"
CFG_LEVELS[globalmax:SCALE]="0 7"
CFG_LEVELS[globalmax:Miranda]="0 8"
CFG_LEVELS[globalmax:S3D]="0 9"
CFG_LEVELS[l1g2]="1 2"
CFG_LEVELS[l2g1]="2 1"
CFG_LEVELS[local5]="5 0"
CFG_LEVELS[local3]="3 0"
CFG_LEVELS[local1]="1 0"

CONFIG_ORDER=(globalmax l1g2 l2g1 local5 local3 local1)

# ── Per-dataset config: data dir (under SDR_ROOT), dtype, mgard dims ───────
declare -A DS_DIR DS_DTYPE DS_DIMS DS_VARS
DS_DIR[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DTYPE[NYX]="s";  DS_DIMS[NYX]="512 512 512"
DS_VARS[NYX]="temperature.f32 velocity_x.f32 velocity_y.f32 velocity_z.f32"

DS_DIR[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500"
DS_DTYPE[Hurricane]="s";  DS_DIMS[Hurricane]="500 500 100"
DS_VARS[Hurricane]="Pf48.bin.f32 Uf48.bin.f32 Vf48.bin.f32 Wf48.bin.f32"

DS_DIR[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200"
DS_DTYPE[SCALE]="s";  DS_DIMS[SCALE]="1200 1200 98"
DS_VARS[SCALE]="PRES-98x1200x1200.f32 T-98x1200x1200.f32 U-98x1200x1200.f32 V-98x1200x1200.f32"

DS_DIR[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384"
DS_DTYPE[Miranda]="d";  DS_DIMS[Miranda]="384 384 256"
DS_VARS[Miranda]="density.d64 diffusivity.d64 pressure.d64 velocityz.d64"

DS_DIR[S3D]="double_precision/SDRBENCH-S3D/sliced"
DS_DTYPE[S3D]="d";  DS_DIMS[S3D]="500 500 500"
DS_VARS[S3D]="CH4.d64 CO2.d64 H2O.d64 O2.d64"

DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)

# ── Tuned relative error bounds ───────────────────────────────────────────
# One entry per (config, dataset); the four values are in DS_VARS order.
# Format: "CONFIG|DATASET|eb_var1|eb_var2|eb_var3|eb_var4"
EB_TABLE=(
  # Global MAX
  "globalmax|NYX|5.39e-3|5.38e-3|5.48e-3|5.31e-3"
  "globalmax|Hurricane|3.76e-3|3.81e-3|3.69e-3|3.31e-3"
  "globalmax|SCALE|3.48e-3|3.48e-3|3.5e-3|3.48e-3"
  "globalmax|Miranda|5.38e-3|5.38e-3|5.38e-3|5.38e-3"
  "globalmax|S3D|4.99e-3|4.99e-3|4.99e-3|4.97e-3"

  # Local 1 + Global 2
  "l1g2|NYX|5.3e-3|5.3e-3|5.2e-3|5.3e-3"
  "l1g2|Hurricane|5.67e-3|5.58e-3|5.67e-3|5.65e-3"
  "l1g2|SCALE|5.35e-3|5.3e-3|5.3e-3|5.28e-3"
  "l1g2|Miranda|5.4e-3|5.4e-3|5.5e-3|5.4e-3"
  "l1g2|S3D|5.3e-3|5.29e-3|5.3e-3|5.5e-3"

  # Local 2 + Global 1
  "l2g1|NYX|5.63e-3|5.3e-3|5.1e-3|5.27e-3"
  "l2g1|Hurricane|5.65e-3|5.45e-3|5.65e-3|5.65e-3"
  "l2g1|SCALE|5.15e-3|5.05e-3|5.05e-3|5.15e-3"
  "l2g1|Miranda|5.4e-3|5.4e-3|5.4e-3|5.4e-3"
  "l2g1|S3D|5.3e-3|5.3e-3|5.3e-3|5.3e-3"

  # Local only, 5 layers
  "local5|NYX|5.3e-3|5.3e-3|5.3e-3|5.3e-3"
  "local5|Hurricane|5.45e-3|5.45e-3|5.45e-3|5.45e-3"
  "local5|SCALE|5.15e-3|5.15e-3|5.15e-3|5.15e-3"
  "local5|Miranda|6.5e-3|6.45e-3|6.5e-3|6.42e-3"
  "local5|S3D|5.3e-3|5.3e-3|5.3e-3|5.3e-3"

  # Local only, 3 layers
  "local3|NYX|5.3e-3|5.25e-3|5.2e-3|5.3e-3"
  "local3|Hurricane|5.3e-3|5.4e-3|5.4e-3|5.4e-3"
  "local3|SCALE|5.05e-3|5.05e-3|4.95e-3|5.05e-3"
  "local3|Miranda|5.4e-3|5.4e-3|5.4e-3|5.4e-3"
  "local3|S3D|5.1e-3|5.09e-3|5.1e-3|5.1e-3"

  # Local only, 1 layer
  "local1|NYX|5.63e-3|5.35e-3|5.3e-3|5.4e-3"
  "local1|Hurricane|5.45e-3|5.45e-3|5.25e-3|5.45e-3"
  "local1|SCALE|5.05e-3|5.05e-3|5.05e-3|5.05e-3"
  "local1|Miranda|5.4e-3|5.4e-3|5.4e-3|5.4e-3"
  "local1|S3D|5.3e-3|5.2e-3|5.3e-3|5.3e-3"
)

# ── Selections ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  SELECTED_DATASETS=("$@")
else
  SELECTED_DATASETS=("${DATASET_ORDER[@]}")
fi

if [[ -n "${CONFIG:-}" ]]; then
  IFS=',' read -r -a SELECTED_CONFIGS <<< "$CONFIG"
else
  SELECTED_CONFIGS=("${CONFIG_ORDER[@]}")
fi

# Raw rows: config,ll,gl,dataset,variable,l_dec,g_dec,l_rec,g_rec
RAW_ROWS=()

# ── Helpers ───────────────────────────────────────────────────────────────
strip_color() { sed -E 's/\x1b\[[0-9;]*m//g'; }

# Levels for a (config, dataset): per-dataset override falls back to default.
levels_for() {
  local cfg=$1 ds=$2
  echo "${CFG_LEVELS[$cfg:$ds]:-${CFG_LEVELS[$cfg]:-}}"
}

# Error bounds (four, space separated) for a (config, dataset).
ebs_for() {
  local cfg=$1 ds=$2 row
  for row in "${EB_TABLE[@]}"; do
    IFS='|' read -r r_cfg r_ds e1 e2 e3 e4 <<< "$row"
    if [[ "$r_cfg" == "$cfg" && "$r_ds" == "$ds" ]]; then
      echo "$e1 $e2 $e3 $e4"
      return 0
    fi
  done
  return 1
}

run_one() {
  local cfg=$1 ds=$2 var=$3 eb=$4 ll=$5 gl=$6
  local in_data="${SDR_ROOT}/${DS_DIR[$ds]}/${var}"
  local dt="${DS_DTYPE[$ds]}"
  local dims="${DS_DIMS[$ds]}"

  echo "=== [${cfg}] ${ds} / ${var}  (-ll ${ll} -gl ${gl}, e=${eb}) ==="
  local cmd=("$EXEC" -z -i "$in_data" -o "$OUT_DATA" -dt "$dt" -dim 3 $dims \
             -em "$ERR_MODE" -e "$eb" $FIXED_FLAGS -ll "$ll" -gl "$gl")

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
    printf '%s\n' "$out" >>"$RUN_LOG"
    echo "  !! command failed — see $RUN_LOG" >&2
    return 1
  }
  printf '%s\n' "$out" >>"$RUN_LOG"

  # "[time] Local Decomposition: 0.004355 s (129.610372 GB/s)" -> field 4.
  # A side the config does not use emits no line; record it as 0.
  local ldec gdec lrec grec
  ldec=$(awk '/\[time\] Local Decomposition:/{print $4; exit}'    <<<"$out")
  gdec=$(awk '/\[time\] Global Decomposition:/{print $4; exit}'   <<<"$out")
  lrec=$(awk '/\[time\] Local Recomposition:/{print $4; exit}'    <<<"$out")
  grec=$(awk '/\[time\] Global Recomposition:/{print $4; exit}'   <<<"$out")

  echo "    -> local dec/rec = ${ldec:-0}/${lrec:-0} s   global dec/rec = ${gdec:-0}/${grec:-0} s"
  RAW_ROWS+=("$cfg,$ll,$gl,$ds,$var,${ldec:-0},${gdec:-0},${lrec:-0},${grec:-0}")
}

# ── Main loop ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
[[ -z "${DRY_RUN:-}" ]] && : > "$RUN_LOG"

for cfg in "${SELECTED_CONFIGS[@]}"; do
  echo "########## Config: ${cfg} ##########"
  for ds in "${SELECTED_DATASETS[@]}"; do
    if [[ -z "${DS_DIR[$ds]:-}" ]]; then
      echo "unknown dataset: $ds (valid: ${DATASET_ORDER[*]})" >&2
      exit 1
    fi

    local_levels=$(levels_for "$cfg" "$ds")
    if [[ -z "$local_levels" ]]; then
      echo "unknown config: $cfg (valid: ${CONFIG_ORDER[*]})" >&2
      exit 1
    fi
    read -r ll gl <<< "$local_levels"

    if ! ebs=$(ebs_for "$cfg" "$ds"); then
      echo "  !! no error bounds for $cfg/$ds, skipping" >&2
      continue
    fi
    read -r -a eb_arr <<< "$ebs"
    read -r -a var_arr <<< "${DS_VARS[$ds]}"

    for i in "${!var_arr[@]}"; do
      run_one "$cfg" "$ds" "${var_arr[$i]}" "${eb_arr[$i]}" "$ll" "$gl"
    done
  done
done

# ── Flush: raw rows + per-dataset averages over the four variables ────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-variable timings (seconds) ==="
    echo "config,ll,gl,dataset,variable,local_decomp_s,global_decomp_s,local_recomp_s,global_recomp_s"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === per-dataset averages over variables (seconds) ==="
    echo "config,ll,gl,dataset,num_variables,avg_local_decomp_s,avg_global_decomp_s,avg_local_recomp_s,avg_global_recomp_s"
    if ((${#RAW_ROWS[@]})); then
      printf '%s\n' "${RAW_ROWS[@]}" | awk -F, '
        {
          key = $1 "," $2 "," $3 "," $4          # config,ll,gl,dataset
          if (!(key in n)) order[++k] = key
          n[key]++; ld[key] += $6; gd[key] += $7; lr[key] += $8; gr[key] += $9
        }
        END {
          for (i = 1; i <= k; i++) {
            key = order[i]
            printf "%s,%d,%.6f,%.6f,%.6f,%.6f\n", key, n[key],
                   ld[key]/n[key], gd[key]/n[key], lr[key]/n[key], gr[key]/n[key]
          }
        }'
    fi
  } > "$RESULTS_FILE"
  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Full tool output:   $RUN_LOG"
fi
