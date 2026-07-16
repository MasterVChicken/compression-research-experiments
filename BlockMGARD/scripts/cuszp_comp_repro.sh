#!/usr/bin/env bash
#
# cuSZp compression baseline — reproduction driver
# ------------------------------------------------
# Data-driven rewrite of cuszp_comp.sh. All experiment parameters live in the
# CONFIG tables below; the runner loops over them so you no longer have to
# comment/uncomment blocks to reproduce a subset of results.
#
# cuSZp takes an ABSOLUTE error bound (-eb abs). For each variable the three
# tuned abs values below correspond to the 1e-2 / 1e-4 / 1e-6 relative levels
# (same labels as the zfp baseline, so the two CSVs line up for plotting).
#
# Usage:
#   ./cuszp_comp_repro.sh                     # run everything
#   ./cuszp_comp_repro.sh NYX Miranda         # only these datasets
#   ERR=1e-4 ./cuszp_comp_repro.sh            # only one error level (1e-2|1e-4|1e-6)
#   ERR=1e-2,1e-6 ./cuszp_comp_repro.sh S3D   # subset of datasets + error levels
#   DRY_RUN=1 ./cuszp_comp_repro.sh           # print the cuSZp commands, don't run
#
set -euo pipefail

# ── Paths / fixed options ────────────────────────────────────────────────
cuszp_exec=/home/leonli/ROITest/comp/install/cuszp/bin/cuSZp
SDR_ROOT=/home/leonli/SDRBENCH
MODE="plain"               # cuSZp -m mode

# Parsed results are written to a single CSV, overwritten on every run. It holds
# three tidy sections (encode time / decode time / ratio), each preceded by a
# "# ===" comment line so you can tell / copy out whichever block you need.
# Every section has columns: dataset,variable,error_bound,<metric>,error_check
# error_check is "pass" or "FAIL" (FAIL = cuSZp's error check did not pass).
# NOTE: cuSZp reports times in ms; they are converted to seconds here so the
# columns match the zfp baseline (encode_time_s / decode_time_s).
RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/cuszp_results.csv}"

# Row buffers, filled during the run and flushed to RESULTS_FILE at the end.
ENC_ROWS=(); DEC_ROWS=(); CR_ROWS=()

# ── Error levels ─────────────────────────────────────────────────────────
# Column order for every abs-eb triple below is: 1e-2  1e-4  1e-6
ALL_ERR=(1e-2 1e-4 1e-6)

# ── Per-dataset config: data dir (under SDR_ROOT), cuSZp type, dims ───────
declare -A DS_DIR DS_TYPE DS_DIMS
DS_DIR[NYX]="single_precision/SDRBENCH-EXASKY-NYX-512x512x512"
DS_TYPE[NYX]="f32";  DS_DIMS[NYX]="512 512 512"

DS_DIR[Hurricane]="single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500"
DS_TYPE[Hurricane]="f32";  DS_DIMS[Hurricane]="500 500 100"

DS_DIR[SCALE]="single_precision/SDRBENCH-SCALE_98x1200x1200"
DS_TYPE[SCALE]="f32";  DS_DIMS[SCALE]="1200 1200 98"

DS_DIR[Miranda]="double_precision/SDRBENCH-Miranda-256x384x384"
DS_TYPE[Miranda]="f64";  DS_DIMS[Miranda]="384 384 256"

DS_DIR[S3D]="double_precision/SDRBENCH-S3D/sliced"
DS_TYPE[S3D]="f64";  DS_DIMS[S3D]="500 500 500"

# Order datasets are run in (also the default "run all" set)
DATASET_ORDER=(NYX Hurricane SCALE Miranda S3D)

# ── Per-variable absolute-error-bound table ──────────────────────────────
# Format:  "DATASET|filename|eb@1e-2|eb@1e-4|eb@1e-6"
# Values are the enabled (non "# #" alternative) abs bounds from cuszp_comp.sh.
RECORDS=(
  # NYX
  "NYX|temperature.f32|47825.835|478.2583|4.7"
  "NYX|velocity_x.f32|504168.5|5040|50"
  "NYX|velocity_y.f32|565056|5650|55"
  "NYX|velocity_z.f32|389379|3890|38"

  # Hurricane
  "Hurricane|Pf48.bin.f32|34.11|0.3411|0.0032"
  "Hurricane|Uf48.bin.f32|0.530225|0.0052|0.00005"
  # NOTE: original marked Vf48/Wf48 as likely failing the error check.
  "Hurricane|Vf48.bin.f32|0.08085793|0.0008085793|0.000008085793"
  "Hurricane|Wf48.bin.f32|0.13333214|0.0013333214|0.000013333214"

  # SCALE
  "SCALE|PRES-98x1200x1200.f32|1018.1|10.18|0.09"
  "SCALE|T-98x1200x1200.f32|3.14|0.0314|0.0031"
  "SCALE|U-98x1200x1200.f32|0.7175|0.0071|0.000069"
  "SCALE|V-98x1200x1200.f32|0.594|0.00593|0.000057"

  # Miranda
  "Miranda|density.d64|0.0300004|0.000300004|0.000003000046"
  "Miranda|diffusivity.d64|0.01666998|0.0001666998|0.000001666998"
  "Miranda|pressure.d64|0.02231449|0.0002231449|0.000002231449"
  "Miranda|velocityz.d64|0.08996110|0.0008996110|0.0000089961"

  # S3D
  "S3D|CH4.d64|0.0003920985|0.000003920985|0.00000003920985"
  "S3D|CO2.d64|0.001|0.00001|0.0000001"
  "S3D|H2O.d64|0.00085|0.0000085|0.000000085"
  "S3D|O2.d64|0.0020805208|0.0000208805208|0.000000208805208"
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

# Map an error label to its abs-eb column index (0/1/2).
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
  local ds=$1 file=$2 err=$3 eb=$4
  local in_data="${SDR_ROOT}/${DS_DIR[$ds]}/${file}"
  local type="${DS_TYPE[$ds]}"
  local dims="${DS_DIMS[$ds]}"

  echo "=== ${ds} / ${file} @ ${err} (eb abs ${eb}) ==="
  local cmd=("$cuszp_exec" -i "$in_data" -t "$type" -m "$MODE" \
             -eb abs "$eb" -d 3 $dims)

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  if [[ ! -f "$in_data" ]]; then
    echo "  !! input not found, skipping: $in_data" >&2
    return 0
  fi

  # Run cuSZp, echo its output, and capture it for parsing.
  # '|| true' so a nonzero exit (e.g. failed check) doesn't abort under set -e.
  local out
  out="$("${cmd[@]}" 2>&1)" || true
  echo "$out"

  # Parse metrics. cuSZp times are in ms -> convert to seconds (/1000).
  #   "cuSZp compression   end-to-end time: 0.814848 ms"
  #   "cuSZp decompression end-to-end time: 0.446016 ms"
  #   "cuSZp compression ratio: 227.694529"
  local enc dec ratio check
  enc=$(awk '/compression +end-to-end time/ && !/decompression/{printf "%.6g", $(NF-1)/1000; exit}' <<<"$out")
  dec=$(awk '/decompression +end-to-end time/{printf "%.6g", $(NF-1)/1000; exit}' <<<"$out")
  ratio=$(awk '/compression ratio:/{print $NF; exit}' <<<"$out")

  # Error check: pass only if cuSZp explicitly says so.
  if grep -q 'Pass error check' <<<"$out"; then
    check="pass"
  else
    check="FAIL"
    echo "  ** WARNING: error check did NOT pass for ${ds}/${file} @ ${err}" >&2
  fi

  echo "    -> encode=${enc:-NA}s  decode=${dec:-NA}s  ratio=${ratio:-NA}  check=${check}"
  ENC_ROWS+=("$ds,$file,$err,${enc:-NA},$check")
  DEC_ROWS+=("$ds,$file,$err,${dec:-NA},$check")
  CR_ROWS+=("$ds,$file,$err,${ratio:-NA},$check")
}

for ds in "${SELECTED_DATASETS[@]}"; do
  if [[ -z "${DS_DIR[$ds]:-}" ]]; then
    echo "unknown dataset: $ds (valid: ${DATASET_ORDER[*]})" >&2
    exit 1
  fi
  for rec in "${RECORDS[@]}"; do
    IFS='|' read -r r_ds r_file e0 e1 e2 <<< "$rec"
    [[ "$r_ds" == "$ds" ]] || continue
    ebs=("$e0" "$e1" "$e2")
    for err in "${SELECTED_ERR[@]}"; do
      idx=$(err_index "$err")
      run_one "$ds" "$r_file" "$err" "${ebs[$idx]}"
    done
  done
done

# ── Flush parsed results to one CSV with three comment-separated sections ──
if [[ -z "${DRY_RUN:-}" ]]; then
  mkdir -p "$(dirname "$RESULTS_FILE")"
  {
    echo "# === encode_time (compression time, seconds) ==="
    echo "dataset,variable,error_bound,encode_time_s,error_check"
    ((${#ENC_ROWS[@]})) && printf '%s\n' "${ENC_ROWS[@]}"
    echo
    echo "# === decode_time (decompression time, seconds) ==="
    echo "dataset,variable,error_bound,decode_time_s,error_check"
    ((${#DEC_ROWS[@]})) && printf '%s\n' "${DEC_ROWS[@]}"
    echo
    echo "# === ratio (compression ratio, CR) ==="
    echo "dataset,variable,error_bound,ratio,error_check"
    ((${#CR_ROWS[@]})) && printf '%s\n' "${CR_ROWS[@]}"
  } > "$RESULTS_FILE"
  echo "Results written to: $RESULTS_FILE"
fi
