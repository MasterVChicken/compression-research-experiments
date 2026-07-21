cmake_minimum_required(VERSION 3.17)
project(adios_io_bench LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── Install directories (adjust if your paths differ) ─────────────────────────
set(MGARD_INSTALL    /home/leonli/MGARD/install-cuda-hopper)
set(ADIOS2_INSTALL   /home/leonli/ROITest/comp/adios2-install)
set(ZFP_INSTALL      /home/leonli/ROITest/comp/install/zfp)
set(CUSZP_INSTALL    /home/leonli/ROITest/comp/install/cuszp)
set(NVCOMP_INSTALL   ${MGARD_INSTALL})  # nvcomp is bundled with MGARD-X

# ── Packages ──────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

find_package(adios2 REQUIRED
    PATHS ${ADIOS2_INSTALL}
    NO_DEFAULT_PATH)

find_package(zfp REQUIRED
    PATHS ${ZFP_INSTALL}
    NO_DEFAULT_PATH)

find_package(CUDAToolkit REQUIRED)

# ── Include directories ────────────────────────────────────────────────────────
include_directories(
    ${MGARD_INSTALL}/include
    ${CUSZP_INSTALL}/include
    ${NVCOMP_INSTALL}/include
)

# ── Library directories ────────────────────────────────────────────────────────
link_directories(
    ${MGARD_INSTALL}/lib
    ${MGARD_INSTALL}/lib64
    ${CUSZP_INSTALL}/lib64
    ${NVCOMP_INSTALL}/lib
)

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(adios_io_bench adios_io_bench.cpp)

target_link_libraries(adios_io_bench
    MPI::MPI_CXX
    adios2::cxx_mpi
    zfp::zfp
    mgard
    cuSZp
    nvcomp
    CUDA::cudart
)

# Suppress CUDA/CXX language mismatch warnings for headers
set_target_properties(adios_io_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CXX
)
