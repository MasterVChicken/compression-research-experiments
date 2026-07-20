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
