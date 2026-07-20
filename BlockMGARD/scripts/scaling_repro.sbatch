#!/bin/bash
#SBATCH --job-name=BlockMGARD_WeakScaling
#SBATCH --partition=chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --account=cdux
#
# BlockMGARD weak-scaling experiment — reproduction driver
# --------------------------------------------------------
# Weak scaling on a single node, 1 -> 4 GPUs. Each GPU independently compresses
# the SAME full set of datasets, so per-GPU work is constant while total work
# grows with the GPU count; ideal weak scaling therefore shows a flat curve and
# any rise exposes contention for shared resources (host memory bandwidth,
# PCIe, page cache, CPU).
#
# The processes do not communicate (no MPI); each is pinned to one GPU via
# CUDA_VISIBLE_DEVICES.
#
# For each scale point (N GPUs):
#   - Warm-up: run all datasets on every allocated GPU in parallel, to fill the
#     page cache and warm up the CUDA contexts (results discarded).
#   - Measure: repeat NREP times; in each repetition N GPUs run all datasets
#     simultaneously. Per dataset, take the MAX kernel time across GPUs (the
#     slowest GPU approximates the makespan), then the MIN across repetitions.
#
# Usage:
#   sbatch scaling_repro.sbatch
#   NREP=5 sbatch scaling_repro.sbatch          # more repetitions
#   GPU_COUNTS="1 4" sbatch scaling_repro.sbatch # only these scale points
#   DRY_RUN=1 ./scaling_repro.sbatch            # print commands, don't run
#
set -uo pipefail

# ── Executable / environment ──────────────────────────────────────────────
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
EXEC=${MGARD_INSTALL}/bin/mgard-x
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib64:${MGARD_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${DRY_RUN:-}" ]]; then
  module load cuda/12.4 gcc/8.5.0
fi

# ── Output locations ──────────────────────────────────────────────────────
RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/scaling_results.csv}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/scaling_logs_${SLURM_JOB_ID:-local}}"

# ── Experiment settings ───────────────────────────────────────────────────
NREP="${NREP:-3}"                       # repetitions per scale point (min taken)
GPU_COUNTS="${GPU_COUNTS:-1 2 3 4}"     # scale points to sweep

# Fixed compression settings. -hh enables the hybrid (block-local) hierarchy,
# which is what -ll/-gl actually configure; without it mgard-x silently falls
# back to the default MultiDim decomposition.
ERR_MODE="rel"
FIXED_FLAGS="-hh -s inf -l huffman -d cuda -ll 1 -gl 2 -v 2"

# ── Datasets: "NAME|file|dtype|dims|error_bound" ──────────────────────────
# Dimensions follow the same order as hybridhierarchy_repro.sh (this experiment
# uses that script's l1g2 configuration, so the two must be comparable).
DATASETS=(
  "NYX_temperature|/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32|s|512 512 512|4.8e-1"
  "Hurricane_Pf48|/home/leonli/SDRBENCH/single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500/Pf48.bin.f32|s|500 500 100|4.36e-1"
  "SCALE_PRES|/home/leonli/SDRBENCH/single_precision/SDRBENCH-SCALE_98x1200x1200/PRES-98x1200x1200.f32|s|1200 1200 98|6.53e-1"
  "Miranda_density|/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/density.d64|d|384 384 256|5.06e-1"
  "S3D_O2|/home/leonli/SDRBENCH/double_precision/SDRBENCH-S3D/sliced/O2.d64|d|500 500 500|4.988e-1"
)
NUM_DATASETS=${#DATASETS[@]}

ds_field() { IFS='|' read -r -a f <<< "${DATASETS[$1]}"; echo "${f[$2]}"; }

# ── Helpers ───────────────────────────────────────────────────────────────
float_max() { awk "BEGIN { print ($1 > $2) ? $1 : $2 }"; }
float_min() { awk "BEGIN { print ($1 < $2) ? $1 : $2 }"; }

# Extract "[time] <Compression|Decompression> Kernel: <val> s" from a log.
# Prints nothing (and returns 1) when the value cannot be found.
parse_kernel() {
  local logfile=$1 label=$2 val
  [[ -f "$logfile" ]] || return 1
  val=$(sed -E 's/\x1b\[[0-9;]*m//g' "$logfile" \
        | awk -v L="$label" '$0 ~ ("\\[time\\] " L " Kernel:") {print $4; exit}')
  [[ -n "$val" ]] || return 1
  printf '%s' "$val"
}

# Run every dataset sequentially on one GPU.
# Usage: run_all_datasets <gpu_id> <slot> <tag> <rep>
run_all_datasets() {
  local gpu_id=$1 slot=$2 tag=$3 rep=$4 j
  for j in $(seq 0 $((NUM_DATASETS - 1))); do
    local name file dt dims eb logfile outfile
    name=$(ds_field "$j" 0); file=$(ds_field "$j" 1)
    dt=$(ds_field "$j" 2);   dims=$(ds_field "$j" 3); eb=$(ds_field "$j" 4)
    logfile="${LOG_DIR}/scale${tag}_rep${rep}_gpu${slot}_${name}.log"
    outfile="${LOG_DIR}/compressed_scale${tag}_gpu${slot}_ds${j}.dat"

    if [[ -n "${DRY_RUN:-}" ]]; then
      echo "  CUDA_VISIBLE_DEVICES=$gpu_id $EXEC -z -i $file -o $outfile" \
           "-dt $dt -dim 3 $dims -em $ERR_MODE -e $eb $FIXED_FLAGS"
      continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id \
      $EXEC -z -i "$file" -o "$outfile" -dt "$dt" -dim 3 $dims \
            -em "$ERR_MODE" -e "$eb" $FIXED_FLAGS \
            > "$logfile" 2>&1
  done
}

# ── Set up ────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
IFS=',' read -ra ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_ALLOCATED=${#ALLOCATED_GPUS[@]}

RAW_ROWS=()     # gpu_count,rep,gpu_slot,dataset,compress_kernel_s,decompress_kernel_s
REP_ROWS=()     # gpu_count,rep,dataset,max_compress_kernel_s,max_decompress_kernel_s
FINAL_ROWS=()   # gpu_count,dataset,min_max_compress_kernel_s,min_max_decompress_kernel_s
PARSE_FAILURES=0

# ── Warm-up: all datasets on every allocated GPU, in parallel ─────────────
echo "=== Warm-up (page cache + CUDA contexts; results discarded) ==="
WARMUP_PIDS=()
for i in $(seq 0 $((NUM_ALLOCATED - 1))); do
  run_all_datasets "${ALLOCATED_GPUS[$i]}" "$i" "warmup" "0" &
  WARMUP_PIDS+=($!)
done
for pid in "${WARMUP_PIDS[@]}"; do wait "$pid"; done
echo "Warm-up complete."

# ── Weak-scaling sweep ────────────────────────────────────────────────────
for GPU_COUNT in $GPU_COUNTS; do
  if (( GPU_COUNT > NUM_ALLOCATED )); then
    echo "!! skipping scale point ${GPU_COUNT}: only ${NUM_ALLOCATED} GPU(s) allocated" >&2
    continue
  fi

  echo
  echo "=========================================="
  echo "Scale point: ${GPU_COUNT} GPU(s)  [$(date)]"
  echo "=========================================="

  declare -a best_comp best_decomp
  for j in $(seq 0 $((NUM_DATASETS - 1))); do
    best_comp[$j]="NA"; best_decomp[$j]="NA"
  done

  for rep in $(seq 1 "$NREP"); do
    echo "  Rep ${rep}/${NREP}..."
    PIDS=()
    for i in $(seq 0 $((GPU_COUNT - 1))); do
      run_all_datasets "${ALLOCATED_GPUS[$i]}" "$i" "$GPU_COUNT" "$rep" &
      PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    [[ -n "${DRY_RUN:-}" ]] && continue

    # Per dataset: MAX across GPUs. A missing value makes the whole
    # repetition unusable for that dataset (recorded as NA, never as 0).
    for j in $(seq 0 $((NUM_DATASETS - 1))); do
      local_name=$(ds_field "$j" 0)
      max_c=""; max_d=""; rep_ok=1

      for i in $(seq 0 $((GPU_COUNT - 1))); do
        logfile="${LOG_DIR}/scale${GPU_COUNT}_rep${rep}_gpu${i}_${local_name}.log"
        c=$(parse_kernel "$logfile" "Compression")   || c=""
        d=$(parse_kernel "$logfile" "Decompression") || d=""

        RAW_ROWS+=("$GPU_COUNT,$rep,$i,$local_name,${c:-NA},${d:-NA}")

        if [[ -z "$c" || -z "$d" ]]; then
          echo "    !! parse failed: $(basename "$logfile") (compress='${c:-}' decompress='${d:-}')" >&2
          PARSE_FAILURES=$((PARSE_FAILURES + 1))
          rep_ok=0
          continue
        fi
        max_c=$(float_max "$c" "${max_c:-0}")
        max_d=$(float_max "$d" "${max_d:-0}")
      done

      if (( rep_ok )) && [[ -n "$max_c" && -n "$max_d" ]]; then
        REP_ROWS+=("$GPU_COUNT,$rep,$local_name,$max_c,$max_d")
        echo "    rep${rep} | ${local_name}: max_comp=${max_c}s  max_decomp=${max_d}s"
        # MIN across repetitions, skipping any NA.
        [[ "${best_comp[$j]}"   == "NA" ]] && best_comp[$j]="$max_c"   || best_comp[$j]=$(float_min   "$max_c" "${best_comp[$j]}")
        [[ "${best_decomp[$j]}" == "NA" ]] && best_decomp[$j]="$max_d" || best_decomp[$j]=$(float_min "$max_d" "${best_decomp[$j]}")
      else
        REP_ROWS+=("$GPU_COUNT,$rep,$local_name,NA,NA")
        echo "    rep${rep} | ${local_name}: NA (incomplete data, excluded)" >&2
      fi
    done
  done

  for j in $(seq 0 $((NUM_DATASETS - 1))); do
    FINAL_ROWS+=("$GPU_COUNT,$(ds_field "$j" 0),${best_comp[$j]},${best_decomp[$j]}")
  done
  unset best_comp best_decomp
done

# ── Flush results ─────────────────────────────────────────────────────────
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-run kernel times (seconds) ==="
    echo "gpu_count,rep,gpu_slot,dataset,compress_kernel_s,decompress_kernel_s"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === per-repetition max across GPUs (seconds) ==="
    echo "gpu_count,rep,dataset,max_compress_kernel_s,max_decompress_kernel_s"
    ((${#REP_ROWS[@]})) && printf '%s\n' "${REP_ROWS[@]}"
    echo
    echo "# === final: min across repetitions of max across GPUs (seconds) ==="
    echo "gpu_count,dataset,min_max_compress_kernel_s,min_max_decompress_kernel_s"
    ((${#FINAL_ROWS[@]})) && printf '%s\n' "${FINAL_ROWS[@]}"
  } > "$RESULTS_FILE"

  echo
  echo "Results written to: $RESULTS_FILE"
  echo "Per-run logs:       $LOG_DIR"
  if (( PARSE_FAILURES > 0 )); then
    echo "!! ${PARSE_FAILURES} parse failure(s) — affected entries are NA, not 0" >&2
  fi
fi
