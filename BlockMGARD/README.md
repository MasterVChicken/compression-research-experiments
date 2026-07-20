# BlockMGARD

Experiments for the BlockMGARD paper.

## Install baselines

Build and install the three baseline compressors (cuZFP, cuSZp, MGARD-X) with
`scripts/install_baselines.sh`. Each tool is cloned from a timing-instrumented
fork, built, and installed under `$WORK_ROOT/install/<tool>`.

```bash
cd scripts
./install_baselines.sh                            # build all three
./install_baselines.sh zfp cuszp                  # build only the named ones (zfp | cuszp | mgard)
WORK_ROOT=/path/to/comp ./install_baselines.sh    # install elsewhere (default: /home/leonli/ROITest/comp)
```

On success it prints the installed executable paths:

```
zfp   : <WORK_ROOT>/install/zfp/bin/zfp
cuSZp : <WORK_ROOT>/install/cuszp/bin/cuSZp
mgard : <WORK_ROOT>/mgard-x/install-cuda-hopper/bin/mgard-x
```

Requires CUDA + CMake. MGARD is currently tested with the Hopper arch via
`build_mgard_cuda_hopper.sh`; override the build job count with `MGARD_JOBS`
(default 32).

## ROI visualization experiments

`scripts/roi_repro.sh` produces the decompressed (`.dec`) outputs used for
the ROI visualization figures, for all three methods (BlockMGARD, cuZFP, cuSZp),
for both experiments:

- `same_quality` — fix visual quality, compare compression ratio (SCALE/PRES)
- `same_cr` — fix compression ratio (~50), compare visual quality (Miranda/density)

Only BlockMGARD uses the ROI tolerance map (built on demand by
`roi_generator/ROIGenerator.cpp`, compiled automatically); the two baselines use
a uniform error bound tuned to match BlockMGARD's operating point.

```bash
cd scripts
./roi_repro.sh                 # both experiments, all methods
./roi_repro.sh same_cr         # one experiment (same_quality | same_cr)
DRY_RUN=1 ./roi_repro.sh       # print commands without running
VERBOSE=1 ./roi_repro.sh       # show tool output live (else it goes to the log)
OUT_DIR=/path ./roi_repro.sh   # where .dec / maps / compressed go
```

Run on a GPU node (the executables need CUDA). Everything is written to
`OUT_DIR` (default `/home/leonli/ROITest/roi_results`):

- one decompressed `.dec` per method and experiment, e.g.
  `blockmgard_scale_pres.dec`, `zfp_scale_pres.dec`, `cuszp_scale_pres.dec`
  (and the `*_miranda_density.dec` set for `same_cr`) — these are the inputs to
  the visualization;
- the ROI tolerance maps and intermediate compressed streams;
- `roi_run.log` — the full console output, including each method's compression
  ratio and BlockMGARD's ROI error-verification summary.

## ROI compression-ratio comparison

`scripts/roi_cr_repro.sh` compares the compression ratio BlockMGARD reaches with
a ROI tolerance map against three uniform-error-bound baselines, on **one
variable per dataset**:

| Method | What it runs |
|--------|--------------|
| `blockmgard_roi` | BlockMGARD with a ROI tolerance map (`-roi -hh`, `-ll 2 -gl 3`) |
| `mgard` | plain MGARD-X at its tuned `1e-4` relative error bound |
| `cuzfp` | cuZFP at its tuned `1e-4` bitrate |
| `cuszp` | cuSZp at its tuned `1e-4` absolute error bound |

The baselines run at the `1e-4` level, which is closest to the ROI region's
tolerance, so the comparison reads as "at the same ROI-region quality, whose
compression ratio is higher". Variables: NYX `velocity_z`, Hurricane `Wf48`,
SCALE `V`, Miranda `velocityz`, S3D `O2`.

Only the compression step runs — mgard-x reports the compression ratio (and, for
the ROI run, the block-wise ROI verification) during `-z`.

```bash
cd scripts
./roi_cr_repro.sh                                  # all datasets, all methods
./roi_cr_repro.sh NYX Miranda                      # only these datasets
METHODS="blockmgard_roi cuzfp" ./roi_cr_repro.sh   # only these methods
DRY_RUN=1 ./roi_cr_repro.sh                        # print commands without running
```

Run on a GPU node. Results go to `results/roi_cr_results.csv`:

```
dataset,variable,method,error_level,compression_ratio
NYX,velocity_z.f32,blockmgard_roi,roi,14.81
NYX,velocity_z.f32,mgard,1e-4,...
NYX,velocity_z.f32,cuzfp,1e-4,...
NYX,velocity_z.f32,cuszp,1e-4,...
```

ROI maps and intermediate compressed streams go to `WORK_DIR` (default
`/home/leonli/ROITest/roi_cr_work`); the full tool output is kept in
`results/roi_cr_run.log`. The console also reports the ROI block-violation count
for each `blockmgard_roi` run, so a mis-applied tolerance map is visible
immediately.

## Local vs. global quantization ablation

`scripts/incacheblock_repro.sh` compares two hierarchy configurations at a fixed
`1e-2` relative error bound, over all five datasets (four variables each):

- `local` — local quantization only (`-ll 1 -gl 0`)
- `global` — global quantization only (`-ll 0 -gl 1`)

For each run it extracts the decomposition and recomposition times matching the
mode (`Local *` for the local runs, `Global *` for the global runs) and averages
them over the four variables of each dataset.

```bash
cd scripts
./incacheblock_repro.sh                 # both modes, all datasets
./incacheblock_repro.sh NYX Miranda     # only these datasets
MODE=local ./incacheblock_repro.sh      # only one mode (local | global)
DRY_RUN=1 ./incacheblock_repro.sh       # print commands without running
```

Run on a GPU node. Results go to `results/incacheblock_results.csv`, which holds
two sections — the raw per-variable timings, and the per-dataset averages:

```
# === per-variable timings (seconds) ===
mode,dataset,variable,decomposition_s,recomposition_s

# === per-dataset averages over variables (seconds) ===
mode,dataset,num_variables,avg_decomposition_s,avg_recomposition_s
```

The full tool output is kept in `results/incacheblock_run.log`.

## Hybrid hierarchy ablation

`scripts/hybridhierarchy_repro.sh` sweeps six local/global refactoring-level
configurations over all five datasets (four variables each), reporting
decomposition and recomposition times averaged over the four variables:

| Config      | `-ll` | `-gl` |
|-------------|-------|-------|
| `globalmax` | 0     | per dataset: NYX 9, Hurricane 7, SCALE 7, Miranda 8, S3D 9 |
| `l1g2`      | 1     | 2     |
| `l2g1`      | 2     | 1     |
| `local5`    | 5     | 0     |
| `local3`    | 3     | 0     |
| `local1`    | 1     | 0     |

Local and global times are recorded separately; whichever side a configuration
does not use (level 0) is recorded as `0`. Every variable has its own tuned
relative error bound, kept in the script's `EB_TABLE`.

```bash
cd scripts
./hybridhierarchy_repro.sh                  # all configs, all datasets
./hybridhierarchy_repro.sh NYX Miranda      # only these datasets
CONFIG=local1 ./hybridhierarchy_repro.sh    # only one config (comma-separated ok)
DRY_RUN=1 ./hybridhierarchy_repro.sh        # print commands without running
```

Run on a GPU node. Timings go to `results/hybridhierarchy_results.csv`, in two
sections:

```
# === per-variable timings (seconds) ===
config,ll,gl,dataset,variable,local_decomp_s,global_decomp_s,local_recomp_s,global_recomp_s

# === per-dataset averages over variables (seconds) ===
config,ll,gl,dataset,num_variables,avg_local_decomp_s,avg_global_decomp_s,avg_local_recomp_s,avg_global_recomp_s
```

Compression ratios go to a separate file, `results/hybridhierarchy_cr.csv`, with
one row per variable (not averaged):

```
config,ll,gl,dataset,variable,compression_ratio
```

The full tool output is kept in `results/hybridhierarchy_run.log`.

## Weak-scaling experiment

`scripts/scaling_repro.sbatch` is a SLURM batch job that measures weak scaling on
a single node from 1 to 4 GPUs, using the `l1g2` configuration (`-ll 1 -gl 2`).

Each GPU independently compresses the **same full set of datasets**, so per-GPU
work stays constant while total work grows with the GPU count — ideal weak
scaling is a flat curve, and any rise exposes contention for shared resources
(host memory bandwidth, PCIe, page cache, CPU). The processes do not
communicate; each is pinned to one GPU via `CUDA_VISIBLE_DEVICES`.

Datasets (one variable each): `NYX_temperature`, `Hurricane_Pf48`, `SCALE_PRES`,
`Miranda_density`, `S3D_O2`.

Methodology per scale point: a warm-up pass (all datasets on every allocated GPU,
results discarded), then `NREP` repetitions. Per dataset it takes the **max**
kernel time across GPUs (the slowest GPU approximates the makespan), then the
**min** across repetitions.

```bash
sbatch scaling_repro.sbatch                    # 1 -> 4 GPUs, 10 repetitions
NREP=20 sbatch scaling_repro.sbatch            # more repetitions
GPU_COUNTS="1 4" sbatch scaling_repro.sbatch   # only these scale points
DRY_RUN=1 ./scaling_repro.sbatch               # print commands without running
```

> **Allocate at least as many CPU cores as GPUs.** The GPUs run concurrently but
> share the job's cores for all host-side work (file reads, H2D/D2H staging,
> memcpy), so a CPU-starved job serialises and the scaling numbers become
> meaningless. The batch header requests `--cpus-per-task=16`; the job prints
> its actual allocation on startup and warns if there are fewer cores than GPUs.

Results go to `results/scaling_results.csv`, in three sections — the raw
per-run times, the per-repetition max across GPUs, and the final aggregate:

```
# === per-run kernel times (seconds) ===
gpu_count,rep,gpu_slot,dataset,compress_kernel_s,decompress_kernel_s

# === per-repetition max across GPUs (seconds) ===
gpu_count,rep,dataset,max_compress_kernel_s,max_decompress_kernel_s

# === final: min across repetitions of max across GPUs (seconds) ===
gpu_count,dataset,min_max_compress_kernel_s,min_max_decompress_kernel_s
```

Values that could not be parsed are recorded as `NA` (never as `0`) and are
excluded from the aggregates; the job prints a warning and a failure count.
Per-run logs are kept under `results/scaling_logs_<jobid>/`.

<!-- ===========================================================================
     PARKED: the zfp / cuSZp baseline sections below are commented out for now
     because those results are not used yet. Uncomment when they are needed.
=========================================================================== -->
<!--

## zfp (cuZFP) baseline

`scripts/zfp_comp_repro.sh` runs the cuZFP baseline over the SDRBENCH datasets.
For each `(dataset, variable, error_bound)` it invokes zfp with a pre-tuned
bitrate that passes the corresponding relative-error check, parses the tool's
output, and collects three metrics: compression time, decompression time, and
compression ratio (CR).

### Setup (one-time)

Edit the paths at the top of the script for your machine:

- `zfp_exec` — path to the zfp binary
- `SDR_ROOT` — root of your local SDRBENCH datasets
- `EXEC_MODE` — backend (`-x cuda` / `-x omp` / `-x serial`), default `-x cuda`

Datasets (SDRBENCH) and the zfp binary are **not** tracked in this repo.

### Run

```bash
cd scripts
./zfp_comp_repro.sh                     # run everything (all datasets, all error levels)
./zfp_comp_repro.sh NYX Miranda         # only these datasets
ERR=1e-4 ./zfp_comp_repro.sh            # only one error level (1e-2 | 1e-4 | 1e-6)
ERR=1e-2,1e-6 ./zfp_comp_repro.sh S3D   # subset of datasets + error levels
DRY_RUN=1 ./zfp_comp_repro.sh           # print the zfp commands without running
```

Datasets: `NYX  Hurricane  SCALE  Miranda  S3D`. Error levels: `1e-2 1e-4 1e-6`.

### Output

Results are written to a single CSV, **overwritten on every run**:

```
scripts/results/zfp_results.csv
```

> The path is relative to the current directory, so run the script from
> `scripts/` (as above) to get `scripts/results/`. Override the location with
> `RESULTS_FILE=/abs/path/results.csv ./zfp_comp_repro.sh`.

The file holds three tidy sections, each preceded by a `# ===` comment line so
you can copy out whichever block you need. Every section has columns
`dataset,variable,error_bound,<metric>`:

```
# === encode_time (compression time, seconds) ===
dataset,variable,error_bound,encode_time_s
NYX,temperature.f32,1e-2,0.00637
...

# === decode_time (decompression time, seconds) ===
dataset,variable,error_bound,decode_time_s
...

# === ratio (compression ratio, CR) ===
dataset,variable,error_bound,ratio
...
```

## cuSZp baseline

`scripts/cuszp_comp_repro.sh` runs the cuSZp baseline over the same datasets.
cuSZp takes an **absolute** error bound (`-eb abs`); for each variable the three
tuned abs values correspond to the `1e-2 / 1e-4 / 1e-6` relative levels, so the
`error_bound` labels line up with the zfp baseline. For each
`(dataset, variable, error_bound)` it invokes cuSZp, parses the output, and
collects compression time, decompression time, compression ratio (CR), and
whether cuSZp's error check passed.

### Setup (one-time)

Edit the paths at the top of the script for your machine:

- `cuszp_exec` — path to the cuSZp binary
- `SDR_ROOT` — root of your local SDRBENCH datasets

Datasets (SDRBENCH) and the cuSZp binary are **not** tracked in this repo.

### Run

```bash
cd scripts
./cuszp_comp_repro.sh                     # run everything (all datasets, all error levels)
./cuszp_comp_repro.sh NYX Miranda         # only these datasets
ERR=1e-4 ./cuszp_comp_repro.sh            # only one error level (1e-2 | 1e-4 | 1e-6)
ERR=1e-2,1e-6 ./cuszp_comp_repro.sh S3D   # subset of datasets + error levels
DRY_RUN=1 ./cuszp_comp_repro.sh           # print the cuSZp commands without running
```

Datasets: `NYX  Hurricane  SCALE  Miranda  S3D`. Error levels: `1e-2 1e-4 1e-6`.

### Output

Results are written to a single CSV, **overwritten on every run**:

```
scripts/results/cuszp_results.csv
```

> The path is relative to the current directory, so run the script from
> `scripts/` (as above) to get `scripts/results/`. Override the location with
> `RESULTS_FILE=/abs/path/results.csv ./cuszp_comp_repro.sh`.

Same three-section layout as the zfp output, but each row has an extra
`error_check` column (`pass` or `FAIL`). **cuSZp times are reported in ms and
converted to seconds here**, so the `encode_time_s` / `decode_time_s` columns
match the zfp baseline. `error_check` is `FAIL` when cuSZp's error check did not
pass — those rows are still recorded but should be treated as not meeting the
target error bound (the script also prints a warning for them).

```
# === encode_time (compression time, seconds) ===
dataset,variable,error_bound,encode_time_s,error_check
NYX,temperature.f32,1e-2,0.000815,pass
...

# === decode_time (decompression time, seconds) ===
dataset,variable,error_bound,decode_time_s,error_check
...

# === ratio (compression ratio, CR) ===
dataset,variable,error_bound,ratio,error_check
...
```

-->
