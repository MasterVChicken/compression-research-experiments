#!/bin/bash
#SBATCH --job-name=io_breakdown
#SBATCH --partition=chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=io_%j.out
#SBATCH --error=io_%j.err
#SBATCH --account=cdux
#
# I/O breakdown on 4 H100 GPUs (reproduces the paper's Fig. 11)
# ---------------------------------------------------
# For every dataset and method, measures the compress -> ADIOS2 write ->
# ADIOS2 read -> decompress pipeline and reports the four times, AVERAGED over
# the four variables of each dataset (as in the paper, Table I / Fig. 11).
#
# Weak scaling: each GPU independently processes the full variable (no
# inter-GPU communication) via `mpirun -n <gpu_count>`. Compression/decompression
# are timed KERNEL-ONLY (data assumed on GPU, transfers excluded) to match the
# paper's methodology; write/read are real ADIOS2 BP I/O.
#
# Methods per the paper: Org.(none), LL(nvcomp-lz4), and cuSZp / cuZFP /
# BlockMGARD at 1e-2 / 1e-4 / 1e-6. cuZFP uses fixed bitrates 4/8/16 bpv.
#
# Usage:
#   sbatch io_repro.sbatch
#   GPU_COUNT=4 sbatch io_repro.sbatch
#   DATASETS="NYX Miranda" sbatch io_repro.sbatch
#   DRY_RUN=1 ./io_repro.sbatch
#
set -uo pipefail

if [[ -z "${DRY_RUN:-}" ]]; then
  module load gcc/8.5.0 cuda/12.4.1 mpi/gcc/8.5.0/mpich/4.1.1
fi

# ── Paths ─────────────────────────────────────────────────────────────────
BENCH_EXEC="${BENCH_EXEC:-/home/leonli/compression-research-experiments/BlockMGARD/io_bench/build/adios_io_bench}"
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
ADIOS2_INSTALL=/home/leonli/ROITest/comp/adios2-install
ZFP_INSTALL=/home/leonli/ROITest/comp/install/zfp
CUSZP_INSTALL=/home/leonli/ROITest/comp/install/cuszp
export LD_LIBRARY_PATH="${MGARD_INSTALL}/lib:${MGARD_INSTALL}/lib64:${ADIOS2_INSTALL}/lib64:${ZFP_INSTALL}/lib64:${CUSZP_INSTALL}/lib64:${LD_LIBRARY_PATH:-}"
export ADIOS2_VERBOSE=0

SDR_ROOT=/home/leonli/SDRBENCH
RESULTS_DIR="${RESULTS_DIR:-results}"
RESULTS_FILE="${RESULTS_FILE:-$RESULTS_DIR/io_results.csv}"
SCRATCH_DIR="${SCRATCH_DIR:-/gpfs/projects/cdux/leonli/io_tmp}"
GPU_COUNT="${GPU_COUNT:-4}"

# ── Per-dataset config: dir | dtype(s/d) | dims("nz ny nx") | variables ────
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

# ── Tuned MGARD rel error bounds:  "ds:var" -> "1e-2 1e-4 1e-6" ────────────
declare -A MGARD_EB
MGARD_EB[NYX:temperature.f32]="6.15e-1 5.34e-3 4.99e-5"
MGARD_EB[NYX:velocity_x.f32]="5.9e-1 5.43e-3 5.45e-5"
MGARD_EB[NYX:velocity_y.f32]="5.68e-1 5.54e-3 5.5e-5"
MGARD_EB[NYX:velocity_z.f32]="5.52e-1 5.51e-3 5.15e-5"
MGARD_EB[Hurricane:Pf48.bin.f32]="4.71e-1 3.78e-3 3.96e-5"
MGARD_EB[Hurricane:Uf48.bin.f32]="3.84e-1 3.82e-3 3.82e-5"
MGARD_EB[Hurricane:Vf48.bin.f32]="3.49e-1 3.82e-3 3.8e-5"
MGARD_EB[Hurricane:Wf48.bin.f32]="3.27e-1 3.55e-3 3e-5"
MGARD_EB[SCALE:PRES-98x1200x1200.f32]="4.86e-1 3.99e-3 3.72e-5"
MGARD_EB[SCALE:T-98x1200x1200.f32]="3.89e-1 3.85e-3 3.52e-5"
MGARD_EB[SCALE:U-98x1200x1200.f32]="4.1e-1 3.87e-3 5e-5"
MGARD_EB[SCALE:V-98x1200x1200.f32]="3.9e-1 3.77e-3 5e-5"
MGARD_EB[Miranda:density.d64]="5.38e-1 5.38e-3 4.98e-5"
MGARD_EB[Miranda:diffusivity.d64]="5.38e-1 5.38e-3 5.2e-5"
MGARD_EB[Miranda:pressure.d64]="5.38e-1 5.38e-3 4.98e-5"
MGARD_EB[Miranda:velocityz.d64]="5.58e-1 5.38e-3 4.8e-5"
MGARD_EB[S3D:CH4.d64]="4.9e-1 4.99e-3 4.79e-5"
MGARD_EB[S3D:CO2.d64]="5.5e-1 5.45e-3 4.9e-5"
MGARD_EB[S3D:H2O.d64]="4.9e-1 4.92e-3 4.79e-5"
MGARD_EB[S3D:O2.d64]="4.9e-1 4.99e-3 4.99e-5"

# ── Tuned cuSZp absolute error bounds:  "ds:var" -> "1e-2 1e-4 1e-6" ───────
declare -A CUSZP_EB
CUSZP_EB[NYX:temperature.f32]="47825.835 478.2583 4.7"
CUSZP_EB[NYX:velocity_x.f32]="504168.5 5040 50"
CUSZP_EB[NYX:velocity_y.f32]="565056 5650 55"
CUSZP_EB[NYX:velocity_z.f32]="389379 3890 38"
CUSZP_EB[Hurricane:Pf48.bin.f32]="34.11 0.3411 0.0032"
CUSZP_EB[Hurricane:Uf48.bin.f32]="0.530225 0.0052 0.00005"
CUSZP_EB[Hurricane:Vf48.bin.f32]="0.08085793 0.0008085793 0.000008085793"
CUSZP_EB[Hurricane:Wf48.bin.f32]="0.13333214 0.0013333214 0.000013333214"
CUSZP_EB[SCALE:PRES-98x1200x1200.f32]="1018.1 10.18 0.09"
CUSZP_EB[SCALE:T-98x1200x1200.f32]="3.14 0.0314 0.0031"
CUSZP_EB[SCALE:U-98x1200x1200.f32]="0.7175 0.0071 0.000069"
CUSZP_EB[SCALE:V-98x1200x1200.f32]="0.594 0.00593 0.000057"
CUSZP_EB[Miranda:density.d64]="0.0300004 0.000300004 0.000003000046"
CUSZP_EB[Miranda:diffusivity.d64]="0.01666998 0.0001666998 0.000001666998"
CUSZP_EB[Miranda:pressure.d64]="0.02231449 0.0002231449 0.000002231449"
CUSZP_EB[Miranda:velocityz.d64]="0.08996110 0.0008996110 0.0000089961"
CUSZP_EB[S3D:CH4.d64]="0.0003920985 0.000003920985 0.00000003920985"
CUSZP_EB[S3D:CO2.d64]="0.001 0.00001 0.0000001"
CUSZP_EB[S3D:H2O.d64]="0.00085 0.0000085 0.000000085"
CUSZP_EB[S3D:O2.d64]="0.0020805208 0.0000208805208 0.000000208805208"

# cuZFP fixed bitrates for 1e-2 / 1e-4 / 1e-6 (bits per value).
ZFP_RATE=(4 8 16)
LEVELS=(1e-2 1e-4 1e-6)

# ── Selections ────────────────────────────────────────────────────────────
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a SELECTED <<< "$DATASETS"
else
  SELECTED=("${DATASET_ORDER[@]}")
fi

mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

RAW_ROWS=()   # dataset,variable,method,level,compress,write,decompress,read

# ── Run one bench invocation, append a raw row ────────────────────────────
# args: dataset variable in_file dtype dims method level compressor tol eb_mode
run_one() {
  local ds=$1 var=$2 in_file=$3 dt=$4 dims=$5 method=$6 level=$7 comp=$8 tol=$9 mode=${10}
  local tag="${ds}_${var%.*}_${method}_${level}"
  local bp="${SCRATCH_DIR}/${tag}.bp"
  local csv="${SCRATCH_DIR}/${tag}.csv"
  local log="${SCRATCH_DIR}/${tag}.log"

  echo "  [${ds}/${var}] ${method} ${level}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "    mpirun -n ${GPU_COUNT} $BENCH_EXEC -i $in_file -c <bp> -o <csv>" \
         "-t $dt -n 3 $dims -m $mode -e $tol -s inf -v data -b 0 -d 0 -p $comp"
    return 0
  fi

  rm -rf "${bp}"* "$csv"
  mpirun -n "$GPU_COUNT" "$BENCH_EXEC" -i "$in_file" -c "$bp" -o "$csv" \
      -t "$dt" -n 3 $dims -m "$mode" -e "$tol" \
      -s inf -v data -b 0 -d 0 -p "$comp" > "$log" 2>&1

  if [[ ! -s "$csv" ]]; then
    echo "    !! failed — see $log" >&2; tail -2 "$log" | sed 's/^/       /' >&2
    RAW_ROWS+=("$ds,$var,$method,$level,NA,NA,NA,NA"); return 0
  fi
  local c w d r
  IFS=',' read -r c _ w _ d _ r _ < "$csv"
  RAW_ROWS+=("$ds,$var,$method,$level,${c:-NA},${w:-NA},${d:-NA},${r:-NA}")
  echo "    -> comp=${c} write=${w} decomp=${d} read=${r}"
}

# ── Main loop ─────────────────────────────────────────────────────────────
for ds in "${SELECTED[@]}"; do
  [[ -n "${DS_DIR[$ds]:-}" ]] || { echo "unknown dataset: $ds" >&2; exit 1; }
  dt="${DS_DT[$ds]}"; dims="${DS_DIMS[$ds]}"
  echo "########## ${ds} (${dt}, dims ${dims}) ##########"

  for var in ${DS_VARS[$ds]}; do
    in_file="${SDR_ROOT}/${DS_DIR[$ds]}/${var}"
    if [[ -z "${DRY_RUN:-}" && ! -f "$in_file" ]]; then
      echo "  !! missing input, skipping: $in_file" >&2; continue
    fi

    # Org. (no compression) and LL (nvcomp lz4) — one point each.
    run_one "$ds" "$var" "$in_file" "$dt" "$dims" none org  0 1e-6 abs
    run_one "$ds" "$var" "$in_file" "$dt" "$dims" nvcomp ll 4 0    abs

    # cuSZp / cuZFP / BlockMGARD at each error level.
    read -r -a m_ebs <<< "${MGARD_EB[$ds:$var]}"
    read -r -a c_ebs <<< "${CUSZP_EB[$ds:$var]}"
    for i in 0 1 2; do
      local_level="${LEVELS[$i]}"
      run_one "$ds" "$var" "$in_file" "$dt" "$dims" cuszp "$local_level" 2 "${c_ebs[$i]}" abs
      run_one "$ds" "$var" "$in_file" "$dt" "$dims" zfp   "$local_level" 3 "${ZFP_RATE[$i]}" abs
      run_one "$ds" "$var" "$in_file" "$dt" "$dims" mgard "$local_level" 1 "${m_ebs[$i]}" rel
    done
  done
  echo
done

# ── Flush: raw rows + per-(dataset,method,level) averages over variables ──
if [[ -z "${DRY_RUN:-}" ]]; then
  {
    echo "# === per-variable (seconds); compress/decompress are kernel-only ==="
    echo "dataset,variable,method,level,compress_s,write_io_s,decompress_s,read_io_s"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}"
    echo
    echo "# === Fig 11: averaged over the four variables per dataset ==="
    echo "dataset,method,level,compress_s,write_io_s,decompress_s,read_io_s,num_vars"
    ((${#RAW_ROWS[@]})) && printf '%s\n' "${RAW_ROWS[@]}" | awk -F, '
      $5 ~ /^[0-9.eE+-]+$/ {
        key=$1","$3","$4
        if(!(key in n)) order[++k]=key
        n[key]++; c[key]+=$5; w[key]+=$6; d[key]+=$7; r[key]+=$8
      }
      END{ for(i=1;i<=k;i++){ key=order[i]
        printf "%s,%.6f,%.6f,%.6f,%.6f,%d\n", key,
               c[key]/n[key], w[key]/n[key], d[key]/n[key], r[key]/n[key], n[key] } }'
  } > "$RESULTS_FILE"
  echo "Results written to: $RESULTS_FILE"
fi
