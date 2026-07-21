# adios_io_bench

Standalone benchmark for the **compress → ADIOS2 write → ADIOS2 read →
decompress** pipeline, comparing BlockMGARD against cuSZp, cuZFP and NVCOMP-LZ4.
Drives the Fig. 11 I/O-breakdown experiment (`scripts/io_repro.sbatch`).

Each MPI rank independently processes the full variable on its own GPU (weak
scaling, no inter-GPU communication); ranks are launched with `mpirun -n <N>`.

## Build

```bash
cd io_bench
./build.sh            # -> build/adios_io_bench
```

Requires (paths are set near the top of `build.sh` and `CMakeLists.txt`):
CUDA, MPI (MPICH), ADIOS2, MGARD-X (the BlockMGARD build), zfp, cuSZp, nvcomp.

`build.sh` already handles two environment quirks on this cluster: it builds
with the MPI compiler wrappers (`mpicc`/`mpicxx`) so `find_package(MPI)`
succeeds, and it puts the gcc-8.5.0 `libatomic` on the link path (mpich's
`libmpi.so` needs it).

## CLI

```
adios_io_bench -i <input> -c <bp base> -o <csv> -t <s|d> -n <ndim d0 d1 d2>
               -m <abs|rel> -e <tol> -s inf -v <var> -b 0 -d 0 -p <compressor>
```
`-p`: `0`=none `1`=mgard(BlockMGARD) `2`=cuSZp `3`=cuZFP(bitrate via `-e`)
`4`=nvcomp-lz4. rank 0 appends one CSV row: `compress,GB/s,write,GB/s,
decompress,GB/s,read,GB/s` (times are the max across ranks).

## What was fixed / non-obvious choices

These are deliberate and needed for correct, paper-comparable numbers — do not
"clean them up" without understanding why:

- **Kernel-only compute timing.** Compression/decompression times are the GPU
  kernel time (data assumed on device), excluding H2D/D2H — matching the paper
  (Section V.B). MGARD is read from its internal `[time] … Kernel` log line; the
  other compressors keep `cudaMalloc`/H2D/D2H *outside* the timer. Timing the
  full host-to-host `mgard_x::compress()` instead inflates MGARD ~30×.
- **Per-rank GPU binding.** `cudaSetDevice(rank % ndev)` in `main`, plus
  `config.dev_id = rank % ndev` for MGARD (which calls `SelectDevice(dev_id)`
  internally and ignores the process-level device). Without this, `mpirun -n N`
  piles every rank onto GPU 0.
- **`config.auto_pin_host_buffers = false`** for MGARD: `cudaHostRegister` takes
  a driver-global lock that serialises across co-located ranks.
- **`MGARD_STD=1`** (env) switches MGARD from BlockMGARD (Hybrid) to standard
  MGARD (MultiDim), for A/B comparison.

The binary lives in `build/` (git-ignored); the source here is what gets tracked.
