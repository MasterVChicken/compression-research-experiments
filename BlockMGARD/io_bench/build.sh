#!/bin/bash
# Build script for adios_io_bench
# Run from: /home/leonli/ROITest/adios_io_bench/

set -e

# ── Load modules ─────────────────────────────────────────────────────────────
module load gcc/8.5.0                   2>/dev/null || true
module load mpi/gcc/8.5.0/mpich/4.1.1   2>/dev/null || true

# Build with the MPI compiler wrappers so find_package(MPI) succeeds. The
# wrappers still drive the loaded gcc/g++ underneath, so ZFP's OpenMP detection
# keeps working.
export CC=$(which mpicc)
export CXX=$(which mpicxx)

# mpich's libmpi.so needs libatomic.so.1 (a gcc-8.5.0 runtime lib) at link time,
# which is not on the default search path. Add it for both the linker
# (LIBRARY_PATH) and the runtime loader (LD_LIBRARY_PATH).
GCC85_LIB=/gpfs/packages/spack/spack-rhel8/opt/spack/linux-rhel8-broadwell/gcc-8.5.0/gcc-8.5.0-3v27uxpkvv2qhf5m6fchven7yocl7v4g/lib64
if [ -e "${GCC85_LIB}/libatomic.so.1" ]; then
    export LIBRARY_PATH="${GCC85_LIB}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${GCC85_LIB}:${LD_LIBRARY_PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ── Install directories (adjust if your paths differ) ────────────────────────
MGARD_INSTALL=/home/leonli/MGARD/install-cuda-hopper
ADIOS2_INSTALL=/home/leonli/ROITest/comp/adios2-install
ZFP_INSTALL=/home/leonli/ROITest/comp/install/zfp
CUSZP_INSTALL=/home/leonli/ROITest/comp/install/cuszp

# ── Set runtime library path ─────────────────────────────────────────────────
export LD_LIBRARY_PATH=${MGARD_INSTALL}/lib:${MGARD_INSTALL}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ADIOS2_INSTALL}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ZFP_INSTALL}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUSZP_INSTALL}/lib64:${LD_LIBRARY_PATH}

# ── Configure & build ────────────────────────────────────────────────────────
mkdir -p "${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_PREFIX_PATH="${MGARD_INSTALL};${ADIOS2_INSTALL};${ZFP_INSTALL};${CUSZP_INSTALL}" \
    -DCMAKE_BUILD_RPATH="${MGARD_INSTALL}/lib;${MGARD_INSTALL}/lib64;${ADIOS2_INSTALL}/lib64;${ZFP_INSTALL}/lib64;${CUSZP_INSTALL}/lib64"

cmake --build "${BUILD_DIR}" --parallel 8

echo ""
echo "Build complete. Binary: ${BUILD_DIR}/adios_io_bench"
