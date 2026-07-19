#!/usr/bin/env bash
#
# Build & install the three baseline compressors (cuZFP, cuSZp, MGARD-X).
# Tidied from comp_baseline.sh. Each tool is cloned (custom timing-instrumented
# forks), configured, built and installed under $WORK_ROOT/install/<tool>.
#
# Usage:
#   ./install_baselines.sh                 # build all three
#   ./install_baselines.sh zfp cuszp       # build only the named ones
#   WORK_ROOT=/path/to/comp ./install_baselines.sh
#
# Valid names: zfp  cuszp  mgard
#
set -euo pipefail

# Baseline working directory (everything is cloned/built/installed under here).
WORK_ROOT="${WORK_ROOT:-/home/leonli/ROITest/comp}"
install_dir="${WORK_ROOT}/install"

# MGARD build jobs (passed to its build script).
MGARD_JOBS="${MGARD_JOBS:-32}"

mkdir -p "${WORK_ROOT}"

# ── cuZFP ─────────────────────────────────────────────────────────────────
zfp_dir="${WORK_ROOT}/zfp"
zfp_src_dir="${zfp_dir}/src"
zfp_build_dir="${zfp_dir}/build"
zfp_install_dir="${install_dir}/zfp"

build_zfp() {
  echo ">>> Building cuZFP"
  [[ -d "${zfp_src_dir}" ]] || \
    git clone -b add_timer https://github.com/MasterVChicken/zfp.git "${zfp_src_dir}"
  mkdir -p "${zfp_build_dir}"
  cmake -S "${zfp_src_dir}" -B "${zfp_build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DZFP_WITH_CUDA=ON \
    -DCMAKE_INSTALL_PREFIX="${zfp_install_dir}"
  cmake --build "${zfp_build_dir}" --config Release
  cmake --install "${zfp_build_dir}"
}

# ── cuSZp (cuSZp3) ────────────────────────────────────────────────────────
cuszp_dir="${WORK_ROOT}/cuszp"
cuszp_src_dir="${cuszp_dir}/src"
cuszp_build_dir="${cuszp_dir}/build"
cuszp_install_dir="${install_dir}/cuszp"

build_cuszp() {
  echo ">>> Building cuSZp"
  [[ -d "${cuszp_src_dir}" ]] || \
    git clone -b add_actual_cr https://github.com/MasterVChicken/cuSZp.git "${cuszp_src_dir}"
  mkdir -p "${cuszp_build_dir}"
  cmake -S "${cuszp_src_dir}" -B "${cuszp_build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${cuszp_install_dir}"
  cmake --build "${cuszp_build_dir}" -j
  cmake --install "${cuszp_build_dir}"
}

# ── MGARD-X ───────────────────────────────────────────────────────────────
# The build script installs to <mgard_x_dir>/install-cuda-hopper (see its
# `install_dir=./install-cuda-hopper`). The executable is named "mgard-x".
mgard_x_dir="${WORK_ROOT}/mgard-x"
mgard_x_install_dir="${mgard_x_dir}/install-cuda-hopper"

build_mgard() {
  echo ">>> Building MGARD-X"
  [[ -d "${mgard_x_dir}" ]] || \
    git clone -b 1.6.0-kernel-time https://github.com/MasterVChicken/MGARD.git "${mgard_x_dir}"
  ( cd "${mgard_x_dir}" && ./build_scripts/build_mgard_cuda_hopper.sh "${MGARD_JOBS}" )
}

# ── Select what to build ──────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(zfp cuszp mgard)
fi

for t in "${TARGETS[@]}"; do
  case "$t" in
    zfp)   build_zfp ;;
    cuszp) build_cuszp ;;
    mgard) build_mgard ;;
    *) echo "unknown target: $t (valid: zfp cuszp mgard)" >&2; exit 1 ;;
  esac
done

# ── Report the installed executables ──────────────────────────────────────
echo
echo "Done. Compressor executables:"
echo "  zfp   : ${zfp_install_dir}/bin/zfp"
echo "  cuSZp : ${cuszp_install_dir}/bin/cuSZp"
echo "  mgard : ${mgard_x_install_dir}/bin/mgard-x"
