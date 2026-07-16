#!/usr/bin/env bash
#
# zfp compression baseline — reproduction driver
# ------------------------------------------------
# Data-driven rewrite of zfp_comp.sh. All experiment parameters live in the
# CONFIG tables below; the runner loops over them so you no longer have to
# comment/uncomment blocks to reproduce a subset of results.
#
# Usage:
#   ./zfp_comp_repro.sh                     # run everything
#   ./zfp_comp_repro.sh NYX Miranda         # only these datasets
#   ERR=1e-4 ./zfp_comp_repro.sh            # only one error level (1e-2|1e-4|1e-6)
#   ERR=1e-2,1e-6 ./zfp_comp_repro.sh S3D   # subset of datasets + error levels
#   DRY_RUN=1 ./zfp_comp_repro.sh           # print the zfp commands, don't run
#
set -euo pipefail

# ── Paths / fixed options ────────────────────────────────────────────────
zfp_exec=/home/leonli/ROITest/comp/install/zfp/bin/zfp
SDR_ROOT=/home/leonli/SDRBENCH
OUTPUT_DIR=/home/leonli/ROITest
OUT_DATA=${OUTPUT_DIR}/compressed.dat
EXEC_MODE="-x cuda"        # backend: -x cuda / -x omp / -x serial
STATS_FLAG="-s"            # -s prints compression stats + error check

# ── Error levels ─────────────────────────────────────────────────────────
# Column order for every rate triple below is: 1e-2  1e-4  1e-6
ALL_ERR=(1e-2 1e-4 1e-6)

# ── Per-dataset config: data dir (under SDR_ROOT), dtype flag, dims ───────
declare -A DS_DIR DS_DTYPE DS_DIMS
DS_DIR[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DTYPE[NYX]="-f";  DS_DIMS[NYX]="512 512 512"

DS_DIR[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500"
DS_DTYPE[Hurricane]="-f";  DS_DIMS[Hurricane]="500 500 100"

DS_DIR[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200"
DS_DTYPE[SCALE]="-f";  DS_DIMS[SCALE]="1200 1200 98"

DS_DIR[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384"
DS_DTYPE[Miranda]="-d";  DS_DIMS[Miranda]="384 384 256"

DS_DIR[S3D]="double_precision/SDRBENCH-S3D/sliced"
DS_DTYPE[S3D]="-d";  DS_DIMS[S3D]="500 500 500"

# Order datasets are run in (also the default "run all" set)
DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)

# ── Per-variable rate table ──────────────────────────────────────────────
# Format:  "DATASET|filename|rate@1e-2|rate@1e-4|rate@1e-6"
RECORDS=(
  # NYX
  "NYX|temperature.f32|4.2|14.1|20.9"
  # "NYX|velocity_x.f32|6.6|13.5|20.1"
  # "NYX|velocity_y.f32|6.9|13.7|20.5"
  # "NYX|velocity_z.f32|7.3|14|20.9"

  # # Hurricane
  # "Hurricane|Pf48.bin.f32|8.5|15.2|21.7"
  # "Hurricane|Uf48.bin.f32|7.5|14.4|21.1"
  # "Hurricane|Vf48.bin.f32|7.9|14.6|21.5"
  # "Hurricane|Wf48.bin.f32|7.6|14.7|21"

  # # SCALE
  # "SCALE|PRES-98x1200x1200.f32|0.9|5.3|9.2"
  # "SCALE|T-98x1200x1200.f32|1.3|8|15"
  # "SCALE|U-98x1200x1200.f32|3.3|10|16.6"
  # "SCALE|V-98x1200x1200.f32|4.3|10.9|17.5"

  # # Miranda
  # "Miranda|density.d64|2.9|9.2|15.9"
  # "Miranda|diffusivity.d64|5|11.9|18.4"
  # "Miranda|pressure.d64|4.4|11.1|17.9"
  # "Miranda|velocityz.d64|2.1|8.5|16"

  # # S3D
  # "S3D|CH4.d64|1.1|10.8|11.2"
  # "S3D|CO2.d64|1.4|4.3|11.1"
  # "S3D|H2O.d64|1.1|4.15|11.5"
  # "S3D|O2.d64|1|10.15|10.4"
)

# ── Resolve selections from args / env ───────────────────────────────────
# Datasets: positional args, else all.
if [[ $# -gt 0 ]]; then
  SELECTED_DATASETS=("$@")
else
  SELECTED_DATASETS=("${DATASET_ORDER[@]}")
fi

# Error levels: ERR env (comma-separated), else all.
if [[ -n "${ERR:-}" ]]; then
  IFS=',' read -r -a SELECTED_ERR <<< "$ERR"
else
  SELECTED_ERR=("${ALL_ERR[@]}")
fi

# Map an error label to its rate-column index (0/1/2).
err_index() {
  case "$1" in
    1e-2) echo 0 ;;
    1e-4) echo 1 ;;
    1e-6) echo 2 ;;
    *) echo "unknown error level: $1" >&2; return 1 ;;
  esac
}

# ── Runner ───────────────────────────────────────────────────────────────
run_one() {
  local ds=$1 file=$2 err=$3 rate=$4
  local in_data="${SDR_ROOT}/${DS_DIR[$ds]}/${file}"
  local dtype="${DS_DTYPE[$ds]}"
  local dims="${DS_DIMS[$ds]}"

  echo "=== ${ds} / ${file} @ ${err} (rate=${rate}) ==="
  local cmd=("$zfp_exec" $EXEC_MODE -i "$in_data" -z "$OUT_DATA" \
             $dtype -3 $dims -r "$rate" $STATS_FLAG)

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  if [[ ! -f "$in_data" ]]; then
    echo "  !! input not found, skipping: $in_data" >&2
    return 0
  fi
  "${cmd[@]}"
}

for ds in "${SELECTED_DATASETS[@]}"; do
  if [[ -z "${DS_DIR[$ds]:-}" ]]; then
    echo "unknown dataset: $ds (valid: ${DATASET_ORDER[*]})" >&2
    exit 1
  fi
  for rec in "${RECORDS[@]}"; do
    IFS='|' read -r r_ds r_file r0 r1 r2 <<< "$rec"
    [[ "$r_ds" == "$ds" ]] || continue
    rates=("$r0" "$r1" "$r2")
    for err in "${SELECTED_ERR[@]}"; do
      idx=$(err_index "$err")
      run_one "$ds" "$r_file" "$err" "${rates[$idx]}"
    done
  done
done
