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

`scripts/roi_experiments.sh` produces the decompressed (`.dec`) outputs used for
the ROI visualization figures, for all three methods (BlockMGARD, cuZFP, cuSZp),
for both experiments:

- `same_quality` — fix visual quality, compare compression ratio (SCALE/PRES)
- `same_cr` — fix compression ratio (~50), compare visual quality (Miranda/density)

Only BlockMGARD uses the ROI tolerance map (built on demand by
`roi_generator/ROIGenerator.cpp`, compiled automatically); the two baselines use
a uniform error bound tuned to match BlockMGARD's operating point.

```bash
cd scripts
./roi_experiments.sh                 # both experiments, all methods
./roi_experiments.sh same_cr         # one experiment (same_quality | same_cr)
DRY_RUN=1 ./roi_experiments.sh       # print commands without running
VERBOSE=1 ./roi_experiments.sh       # show tool output live (else it goes to the log)
OUT_DIR=/path ./roi_experiments.sh   # where .dec / maps / compressed go
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
