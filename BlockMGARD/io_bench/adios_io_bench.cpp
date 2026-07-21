/*
 * ADIOS2 I/O Benchmark with External Compressors (4-GPU MPI version)
 *
 * Compressor types (-p):
 *   0 = no compression (Org. I/O baseline)
 *   1 = MGARD-X (lossy, relative error)
 *   2 = cuSZp   (lossy, absolute error)
 *   3 = ZFP-CUDA (lossy, bitrate mode - bits per value)
 *   4 = nvcomp-LZ4 (lossless)
 *
 * Usage:
 *   mpirun -n 4 adios_io_bench -i <input.bin> -c <output.bp> -o <result.csv>
 *                  -t <s|d> -n <ndim> <d0> <d1> ... -m <abs|rel>
 *                  -e <tol> -s inf -v <varname> -b 0 -d 0 -p <type>
 *
 * Output CSV (appended, rank 0 only):
 *   compress_time, compress_GB/s, write_io_time, write_GB/s,
 *   decompress_time, decompress_GB/s, read_io_time, read_GB/s
 *
 * NOTE: Each MPI rank writes/reads its own independent BP file
 *   (<output.bp>.rankN). All ranks compress the full dataset
 *   independently. Timings are wall-clock (slowest rank via Barrier).
 */

#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <functional>
#include <unistd.h>

#include <adios2.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "mgard/compress_x.hpp"
#include "mgard/mgard-x/RuntimeX/Utilities/Timer.hpp"

#include "cuSZp.h"

#include "zfp.h"

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#define OUTPUT_SAFETY_OVERHEAD 1.5

/* ------------------------------------------------------------------ */
/* Kernel-only timing helpers                                          */
/*                                                                     */
/* To match the paper's methodology (Section V.B: "only kernel         */
/* execution time, excluding data transfer"), MGARD compute is taken   */
/* from its internal "[time] ... Kernel" line rather than the full     */
/* host-to-host mgard_x::compress() wall clock (which includes H2D).   */
/* ------------------------------------------------------------------ */

// Run fn() while capturing everything it writes to stdout; return it.
static std::string capture_stdout(const std::function<void()> &fn) {
  fflush(stdout);
  int saved = dup(STDOUT_FILENO);
  char tmpl[] = "/tmp/iobench_cap_XXXXXX";
  int fd = mkstemp(tmpl);
  dup2(fd, STDOUT_FILENO);
  fn();
  fflush(stdout);
  dup2(saved, STDOUT_FILENO);
  close(saved);
  lseek(fd, 0, SEEK_SET);
  std::string out;
  char buf[8192];
  ssize_t n;
  while ((n = read(fd, buf, sizeof(buf))) > 0) out.append(buf, (size_t)n);
  close(fd);
  unlink(tmpl);
  return out;
}

// Parse the seconds value from a line like:
//   "[time] Compression Kernel: 0.019377 s (29.13 GB/s)"
// Returns -1 if the label is not found.
static double parse_time_line(const std::string &log, const std::string &label) {
  size_t p = log.find(label);
  if (p == std::string::npos) return -1.0;
  p = log.find(':', p);
  if (p == std::string::npos) return -1.0;
  return std::strtod(log.c_str() + p + 1, nullptr);
}

using namespace std::chrono;

/* ------------------------------------------------------------------ */
/* Argument parsing helpers                                            */
/* ------------------------------------------------------------------ */

static void print_usage() {
  printf(
    "adios_io_bench: ADIOS2 I/O benchmark with external compressors (4-GPU MPI)\n"
    "Options:\n"
    "  -i <input raw binary file>\n"
    "  -c <compressed ADIOS2 BP output file (base name; .rankN appended per rank)>\n"
    "  -o <CSV result file (append mode)>\n"
    "  -t <s|d>           data type: s=float, d=double\n"
    "  -n <ndim> <d0> ... number of dims followed by sizes\n"
    "  -m <abs|rel>       error bound mode\n"
    "  -e <tol>           error / bitrate value\n"
    "  -s <smoothness>    s parameter for MGARD (use inf)\n"
    "  -v <varname>       variable name in ADIOS2\n"
    "  -b <step_start>    start step (use 0)\n"
    "  -d <step_end>      end step   (use 0)\n"
    "  -p <compressor>    0=none 1=mgard 2=cuszp 3=zfp 4=nvcomp-lz4\n"
  );
}

static bool has_arg(int argc, char *argv[], const std::string &opt) {
  for (int i = 0; i < argc; i++)
    if (opt == argv[i]) return true;
  return false;
}

static std::string get_arg(int argc, char *argv[], const std::string &opt) {
  for (int i = 0; i < argc - 1; i++)
    if (opt == argv[i]) return argv[i + 1];
  printf("Missing argument: %s\n", opt.c_str()); print_usage(); exit(1);
}

static int get_arg_int(int argc, char *argv[], const std::string &opt) {
  return std::stoi(get_arg(argc, argv, opt));
}

static double get_arg_double(int argc, char *argv[], const std::string &opt) {
  return std::stod(get_arg(argc, argv, opt));
}

static std::vector<size_t> get_arg_dims(int argc, char *argv[],
                                         const std::string &opt) {
  for (int i = 0; i < argc - 1; i++) {
    if (opt == argv[i]) {
      int ndim = std::stoi(argv[i + 1]);
      std::vector<size_t> shape(ndim);
      for (int d = 0; d < ndim; d++)
        shape[d] = std::stoul(argv[i + 2 + d]);
      return shape;
    }
  }
  printf("Missing argument: %s\n", opt.c_str()); print_usage(); exit(1);
}

/* ------------------------------------------------------------------ */
/* Compressor: none                                                    */
/* ------------------------------------------------------------------ */

template <typename T>
void no_compress(T *src, void *dst, size_t &cmp_size, size_t n) {
  memcpy(dst, src, n * sizeof(T));
  cmp_size = n * sizeof(T);
}

void no_decompress(void *src, size_t cmp_size, void *dst) {
  memcpy(dst, src, cmp_size);
}

/* ------------------------------------------------------------------ */
/* Compressor: MGARD-X (BlockMGARD: hybrid block-local hierarchy)      */
/* ------------------------------------------------------------------ */

// BlockMGARD refactoring levels. Compression and decompression MUST use the
// same values or the recomposition hierarchy will not match the decomposition.
static constexpr int BLOCKMGARD_LOCAL_LEVELS  = 1;  // -ll
static constexpr int BLOCKMGARD_GLOBAL_LEVELS = 2;  // -gl

template <typename T>
// Returns compressed buffer (caller must free()) and sets cmp_size
double mgard_compress(T *original, void *&compressed, size_t &cmp_size,
                      std::vector<size_t> shape, double tol, double s,
                      const std::string &eb_mode) {
  mgard_x::error_bound_type mode =
      (eb_mode == "abs") ? mgard_x::error_bound_type::ABS
                         : mgard_x::error_bound_type::REL;
  mgard_x::data_type dtype = std::is_same<T, double>::value
                                 ? mgard_x::data_type::Double
                                 : mgard_x::data_type::Float;
  mgard_x::Config config;
  config.dev_type = mgard_x::device_type::CUDA;
  config.lossless = mgard_x::lossless_type::Huffman;
  // MGARD_STD=1 -> standard MGARD (MultiDim). Default -> BlockMGARD (Hybrid).
  if (getenv("MGARD_STD")) {
    config.decomposition = mgard_x::decomposition_type::MultiDim;
  } else {
    config.decomposition = mgard_x::decomposition_type::Hybrid;  // Block MGARD
    config.num_local_refactoring_level  = BLOCKMGARD_LOCAL_LEVELS;
    config.num_global_refactoring_level = BLOCKMGARD_GLOBAL_LEVELS;
  }
  // cudaHostRegister takes a driver-global lock and serialises across the
  // co-located ranks, so it inflates compress time as the GPU count grows.
  // Disable it for a fair multi-GPU comparison (H2D falls back to pageable).
  config.auto_pin_host_buffers = false;
  // MGARD-X calls SelectDevice(config.dev_id) internally, overriding the
  // process-level cudaSetDevice. Point it at this rank's GPU, or every rank
  // piles onto GPU 0 and OOMs under `mpirun -n N`.
  {
    int mrank = 0, ndev = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) config.dev_id = mrank % ndev;
  }
  // Enable timing logs so we can read the internal kernel time.
  config.log_level = mgard_x::log::TIME;

  mgard_x::DIM D = static_cast<mgard_x::DIM>(shape.size());
  std::vector<mgard_x::SIZE> mshape(shape.begin(), shape.end());

  compressed = nullptr;  // MGARD-X allocates this
  cmp_size   = 0;

  mgard_x::compress_status_type ret;
  MPI_Barrier(MPI_COMM_WORLD);
  std::string log = capture_stdout([&]() {
    ret = mgard_x::compress(D, dtype, mshape, tol, s, mode, original,
                            compressed, cmp_size, config, false);
  });
  MPI_Barrier(MPI_COMM_WORLD);
  if (ret != mgard_x::compress_status_type::Success) {
    fprintf(stderr, "MGARD-X compression failed\n"); exit(1);
  }

  // Kernel-only compute (Decompose + Quantize + LosslessCompress), no H2D.
  double t = parse_time_line(log, "Compression Kernel");
  return t >= 0 ? t : 0.0;
}

double mgard_decompress(void *compressed, size_t cmp_size, void *&decompressed) {
  mgard_x::Config config;
  config.dev_type = mgard_x::device_type::CUDA;
  config.lossless = mgard_x::lossless_type::Huffman;
  if (getenv("MGARD_STD")) {
    config.decomposition = mgard_x::decomposition_type::MultiDim;
  } else {
    config.decomposition = mgard_x::decomposition_type::Hybrid;  // Block MGARD
    config.num_local_refactoring_level  = BLOCKMGARD_LOCAL_LEVELS;
    config.num_global_refactoring_level = BLOCKMGARD_GLOBAL_LEVELS;
  }
  // See mgard_compress: pinning serialises across co-located ranks.
  config.auto_pin_host_buffers = false;
  // See mgard_compress: bind MGARD to this rank's GPU (it ignores cudaSetDevice).
  {
    int mrank = 0, ndev = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) config.dev_id = mrank % ndev;
  }
  config.log_level = mgard_x::log::TIME;

  decompressed = nullptr;  // MGARD-X allocates this

  MPI_Barrier(MPI_COMM_WORLD);
  std::string log = capture_stdout([&]() {
    mgard_x::decompress(compressed, cmp_size, decompressed, config, false);
  });
  MPI_Barrier(MPI_COMM_WORLD);

  // Kernel-only compute (LosslessDecompress + Dequantize + Recompose), no D2H.
  double t = parse_time_line(log, "Decompression Kernel");
  return t >= 0 ? t : 0.0;
}

/* ------------------------------------------------------------------ */
/* Compressor: cuSZp                                                   */
/* ------------------------------------------------------------------ */

template <typename T>
double cuszp_compress(T *original, unsigned char *compressed,
                      size_t &cmp_size, std::vector<size_t> shape,
                      double tol) {
  size_t n = 1;
  for (auto d : shape) n *= d;

  // Device buffers + H2D are outside the timer (kernel-only measurement).
  T *d_original;
  unsigned char *d_compressed;
  cudaMalloc(&d_original, n * sizeof(T));
  cudaMalloc(&d_compressed, n * sizeof(T) * OUTPUT_SAFETY_OVERHEAD);
  cudaMemcpy(d_original, original, n * sizeof(T), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cuszp_type_t ctype = std::is_same<T, float>::value
                           ? CUSZP_TYPE_FLOAT : CUSZP_TYPE_DOUBLE;
  uint3 dims3 = {(unsigned)shape[2], (unsigned)shape[1], (unsigned)shape[0]};

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  cuSZp_compress(d_original, d_compressed, n, &cmp_size, (float)tol,
                 CUSZP_DIM_3D, dims3, ctype, CUSZP_MODE_PLAIN, stream);
  cudaStreamSynchronize(stream);
  timer.end();

  cudaMemcpy(compressed, d_compressed, cmp_size, cudaMemcpyDeviceToHost);
  cudaFree(d_original);
  cudaFree(d_compressed);
  cudaStreamDestroy(stream);
  return timer.get();
}

template <typename T>
double cuszp_decompress(unsigned char *compressed, size_t cmp_size,
                        T *decompressed, std::vector<size_t> shape,
                        double tol) {
  size_t n = 1;
  for (auto d : shape) n *= d;

  unsigned char *d_compressed;
  T *d_decompressed;
  cudaMalloc(&d_compressed, cmp_size);
  cudaMalloc(&d_decompressed, n * sizeof(T));
  cudaMemcpy(d_compressed, compressed, cmp_size, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cuszp_type_t ctype = std::is_same<T, float>::value
                           ? CUSZP_TYPE_FLOAT : CUSZP_TYPE_DOUBLE;
  uint3 dims3 = {(unsigned)shape[2], (unsigned)shape[1], (unsigned)shape[0]};

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  cuSZp_decompress(d_decompressed, d_compressed, n, cmp_size, (float)tol,
                   CUSZP_DIM_3D, dims3, ctype, CUSZP_MODE_PLAIN, stream);
  cudaStreamSynchronize(stream);
  timer.end();

  cudaMemcpy(decompressed, d_decompressed, n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaStreamDestroy(stream);
  return timer.get();
}

/* ------------------------------------------------------------------ */
/* Compressor: ZFP-CUDA (bitrate mode)                                */
/* ------------------------------------------------------------------ */

template <typename T>
double zfp_compress_cuda(T *original, uint8_t *compressed, size_t &cmp_size,
                         std::vector<size_t> shape, double bitrate) {
  size_t n = 1;
  for (auto d : shape) n *= d;
  zfp_type type = std::is_same<T, double>::value ? zfp_type_double : zfp_type_float;

  // Input on device; H2D outside the timer (kernel-only measurement).
  T *d_original;
  cudaMalloc(&d_original, n * sizeof(T));
  cudaMemcpy(d_original, original, n * sizeof(T), cudaMemcpyHostToDevice);

  zfp_field *field = nullptr;
  if (shape.size() == 3)
    field = zfp_field_3d(d_original, type, shape[2], shape[1], shape[0]);
  else if (shape.size() == 2)
    field = zfp_field_2d(d_original, type, shape[1], shape[0]);
  else
    field = zfp_field_1d(d_original, type, shape[0]);

  zfp_stream *zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, bitrate, type, zfp_field_dimensionality(field), zfp_false);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  void *d_stream;
  cudaMalloc(&d_stream, bufsize);
  bitstream *bs = stream_open(d_stream, bufsize);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    fprintf(stderr, "ZFP CUDA execution not available\n"); exit(1);
  }

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  cmp_size = zfp_compress(zfp, field);
  cudaDeviceSynchronize();
  timer.end();
  if (cmp_size == 0) {
    fprintf(stderr, "ZFP compression failed\n"); exit(1);
  }

  cudaMemcpy(compressed, d_stream, cmp_size, cudaMemcpyDeviceToHost);
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  cudaFree(d_original);
  cudaFree(d_stream);
  return timer.get();
}

template <typename T>
double zfp_decompress_cuda(uint8_t *compressed, size_t cmp_size,
                           T *decompressed, std::vector<size_t> shape,
                           double bitrate) {
  size_t n = 1;
  for (auto d : shape) n *= d;
  zfp_type type = std::is_same<T, double>::value ? zfp_type_double : zfp_type_float;

  // Compressed stream + output on device; transfers outside the timer.
  void *d_stream;
  T *d_decompressed;
  cudaMalloc(&d_stream, cmp_size);
  cudaMalloc(&d_decompressed, n * sizeof(T));
  cudaMemcpy(d_stream, compressed, cmp_size, cudaMemcpyHostToDevice);

  zfp_field *field = nullptr;
  if (shape.size() == 3)
    field = zfp_field_3d(d_decompressed, type, shape[2], shape[1], shape[0]);
  else if (shape.size() == 2)
    field = zfp_field_2d(d_decompressed, type, shape[1], shape[0]);
  else
    field = zfp_field_1d(d_decompressed, type, shape[0]);

  zfp_stream *zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, bitrate, type, zfp_field_dimensionality(field), zfp_false);

  bitstream *bs = stream_open(d_stream, cmp_size);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    fprintf(stderr, "ZFP CUDA execution not available\n"); exit(1);
  }

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  int ok = zfp_decompress(zfp, field);
  cudaDeviceSynchronize();
  timer.end();
  if (!ok) {
    fprintf(stderr, "ZFP decompression failed\n"); exit(1);
  }

  cudaMemcpy(decompressed, d_decompressed, n * sizeof(T), cudaMemcpyDeviceToHost);
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  cudaFree(d_stream);
  cudaFree(d_decompressed);
  return timer.get();
}

/* ------------------------------------------------------------------ */
/* Compressor: nvcomp-LZ4                                             */
/* ------------------------------------------------------------------ */

template <typename T>
double nvcomp_lz4_compress(T *original, uint8_t *compressed, size_t &cmp_size,
                            size_t n) {
  T *d_original;
  uint8_t *d_compressed;
  cudaMalloc(&d_original, n * sizeof(T));
  cudaMalloc(&d_compressed, cmp_size);
  cudaMemcpy(d_original, original, n * sizeof(T), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t chunk_size = 1 << 15;
  nvcomp::LZ4Manager nvcomp_mgr{chunk_size, NVCOMP_TYPE_UCHAR, stream};
  size_t input_count = n * sizeof(T);
  auto comp_config = nvcomp_mgr.configure_compression(input_count);

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  nvcomp_mgr.compress((uint8_t *)d_original, d_compressed, comp_config);
  cudaStreamSynchronize(stream);
  timer.end();

  cmp_size = nvcomp_mgr.get_compressed_output_size(d_compressed);
  cudaMemcpy(compressed, d_compressed, cmp_size, cudaMemcpyDeviceToHost);
  cudaFree(d_original);
  cudaFree(d_compressed);
  cudaStreamDestroy(stream);
  return timer.get();
}

template <typename T>
double nvcomp_lz4_decompress(uint8_t *compressed, size_t cmp_size,
                              T *decompressed, size_t n) {
  uint8_t *d_compressed;
  T *d_decompressed;
  cudaMalloc(&d_compressed, cmp_size);
  cudaMalloc(&d_decompressed, n * sizeof(T));
  cudaMemcpy(d_compressed, compressed, cmp_size, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto decomp_mgr = nvcomp::create_manager(d_compressed, stream);
  auto decomp_config = decomp_mgr->configure_decompression(d_compressed);

  mgard_x::Timer timer;
  MPI_Barrier(MPI_COMM_WORLD);
  timer.start();
  decomp_mgr->decompress((uint8_t *)d_decompressed, d_compressed, decomp_config);
  cudaStreamSynchronize(stream);
  timer.end();

  cudaMemcpy(decompressed, d_decompressed, n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaStreamDestroy(stream);
  return timer.get();
}

/* ------------------------------------------------------------------ */
/* Main benchmark driver                                               */
/* ------------------------------------------------------------------ */

template <typename T>
void run_benchmark(const std::string &input_file, const std::string &output_bp,
                   const std::string &log_csv, const std::string &var_name,
                   int step_start, int step_end,
                   std::vector<size_t> shape, double tol, double s,
                   const std::string &eb_mode, int compressor_type) {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  size_t n = 1;
  for (auto d : shape) n *= d;

  /* read raw binary data into host buffer */
  std::vector<T> data(n);
  if (rank == 0) {
    std::ifstream f(input_file, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", input_file.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
    f.read(reinterpret_cast<char *>(data.data()), n * sizeof(T));
  }
  MPI_Bcast(data.data(), n * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

  /* allocate compression buffer (non-MGARD compressors) */
  size_t cmp_buf_size = (size_t)(n * sizeof(T) * OUTPUT_SAFETY_OVERHEAD);
  std::vector<uint8_t> cmp_buf(cmp_buf_size);

  /* ---- COMPRESSION ---- */
  double compress_time = 0.0;
  size_t cmp_size = cmp_buf_size;

  // For MGARD, output is allocated by MGARD-X itself (output_pre_allocated=false)
  void *mgard_cmp_buf = nullptr;

  if (compressor_type == 0) {
    no_compress(data.data(), cmp_buf.data(), cmp_size, n);
  } else if (compressor_type == 1) {
    compress_time = mgard_compress(data.data(), mgard_cmp_buf, cmp_size,
                                   shape, tol, s, eb_mode);
  } else if (compressor_type == 2) {
    compress_time = cuszp_compress(data.data(),
                                   reinterpret_cast<unsigned char *>(cmp_buf.data()),
                                   cmp_size, shape, tol);
  } else if (compressor_type == 3) {
    compress_time = zfp_compress_cuda(data.data(), cmp_buf.data(), cmp_size,
                                      shape, tol);
  } else if (compressor_type == 4) {
    compress_time = nvcomp_lz4_compress(data.data(), cmp_buf.data(), cmp_size, n);
  }

  // Unified pointer for ADIOS2 write
  const uint8_t *write_ptr = (compressor_type == 1)
      ? reinterpret_cast<const uint8_t *>(mgard_cmp_buf)
      : cmp_buf.data();

  /* ---- WRITE: each rank writes its own independent BP file ---- */
  // Rank-specific file: output_bp.rank0, output_bp.rank1, ...
  std::string rank_bp = output_bp + ".rank" + std::to_string(rank);

  // Use MPI_COMM_SELF so each rank's ADIOS2 instance is fully independent
  adios2::ADIOS adios(MPI_COMM_SELF);
  adios2::IO write_io = adios.DeclareIO("Write");
  write_io.SetEngine("BP5");

  using C = unsigned char;
  adios2::Dims global_shape = {cmp_size};
  adios2::Dims local_start  = {0};
  adios2::Dims local_count  = {cmp_size};
  adios2::Variable<C> cmp_var =
      write_io.DefineVariable<C>(var_name, global_shape, local_start, local_count);

  adios2::Engine writer = write_io.Open(rank_bp, adios2::Mode::Write);

  mgard_x::Timer write_timer;
  MPI_Barrier(MPI_COMM_WORLD);
  write_timer.start();
  writer.BeginStep();
  writer.Put<C>(cmp_var, write_ptr, adios2::Mode::Sync);
  MPI_Barrier(MPI_COMM_WORLD);
  writer.EndStep();
  write_timer.end();
  double write_io_time = write_timer.get();
  writer.Close();

  /* ---- READ: each rank reads back its own BP file ---- */
  adios2::IO read_io = adios.DeclareIO("Read");
  read_io.SetEngine("BP5");
  adios2::Engine reader = read_io.Open(rank_bp, adios2::Mode::Read);

  std::vector<C> read_buf;

  mgard_x::Timer read_timer;
  MPI_Barrier(MPI_COMM_WORLD);
  read_timer.start();
  reader.BeginStep();
  adios2::Variable<C> read_var = read_io.InquireVariable<C>(var_name);
  adios2::Dims read_start = {0};
  adios2::Dims read_count = {cmp_size};
  read_var.SetSelection({read_start, read_count});
  reader.Get<C>(read_var, read_buf, adios2::Mode::Sync);
  MPI_Barrier(MPI_COMM_WORLD);
  reader.EndStep();
  read_timer.end();
  double read_io_time = read_timer.get();
  reader.Close();

  // Free MGARD compress buffer now that ADIOS2 write is done
  if (compressor_type == 1 && mgard_cmp_buf) { free(mgard_cmp_buf); mgard_cmp_buf = nullptr; }

  /* ---- DECOMPRESSION ---- */
  double decompress_time = 0.0;

  if (compressor_type == 0) {
    std::vector<T> dec_data(n);
    no_decompress(read_buf.data(), cmp_size, dec_data.data());
  } else if (compressor_type == 1) {
    void *mgard_dec_buf = nullptr;
    decompress_time = mgard_decompress(read_buf.data(), cmp_size, mgard_dec_buf);
    free(mgard_dec_buf);
  } else if (compressor_type == 2) {
    std::vector<T> dec_data(n);
    decompress_time = cuszp_decompress(
        reinterpret_cast<unsigned char *>(read_buf.data()),
        cmp_size, dec_data.data(), shape, tol);
  } else if (compressor_type == 3) {
    std::vector<T> dec_data(n);
    decompress_time = zfp_decompress_cuda(read_buf.data(), cmp_size,
                                          dec_data.data(), shape, tol);
  } else if (compressor_type == 4) {
    std::vector<T> dec_data(n);
    decompress_time = nvcomp_lz4_decompress(read_buf.data(), cmp_size,
                                             dec_data.data(), n);
  }

  /* ---- write CSV results (rank 0 only, timings = slowest rank via Barrier) ---- */
  if (rank == 0) {
    double data_GB = (double)(n * sizeof(T)) / 1e9;
    double cmp_throughput  = (compress_time > 0)   ? data_GB / compress_time  : 0;
    double dec_throughput  = (decompress_time > 0) ? data_GB / decompress_time : 0;
    double write_throughput = data_GB / (compress_time + write_io_time);
    double read_throughput  = data_GB / (decompress_time + read_io_time);

    std::ofstream csv(log_csv, std::ios::app);
    csv << compress_time   << "," << cmp_throughput  << ","
        << write_io_time   << "," << write_throughput << ","
        << decompress_time << "," << dec_throughput  << ","
        << read_io_time    << "," << read_throughput  << "\n";
  }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Bind each rank to its own GPU. Works whether all GPUs are visible
  // (rank % ndev spreads them) or each rank sees only one (ndev==1 -> 0).
  // Without this, an `mpirun -n N` job piles every rank onto GPU 0.
  {
    int mrank = 0, ndev = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    cudaGetDeviceCount(&ndev);
    if (ndev > 0) cudaSetDevice(mrank % ndev);
  }

  if (!has_arg(argc, argv, "-i")) { print_usage(); MPI_Finalize(); return 1; }

  std::string input_file = get_arg(argc, argv, "-i");
  std::string output_bp  = get_arg(argc, argv, "-c");
  std::string log_csv    = get_arg(argc, argv, "-o");
  std::string var_name   = get_arg(argc, argv, "-v");
  std::string dt         = get_arg(argc, argv, "-t");
  std::string eb_mode    = get_arg(argc, argv, "-m");
  double tol             = get_arg_double(argc, argv, "-e");
  double s               = get_arg_double(argc, argv, "-s");
  int step_start         = get_arg_int(argc, argv, "-b");
  int step_end           = get_arg_int(argc, argv, "-d");
  int compressor_type    = get_arg_int(argc, argv, "-p");
  std::vector<size_t> shape = get_arg_dims(argc, argv, "-n");

  if (dt == "s") {
    run_benchmark<float>(input_file, output_bp, log_csv, var_name,
                         step_start, step_end, shape, tol, s, eb_mode,
                         compressor_type);
  } else if (dt == "d") {
    run_benchmark<double>(input_file, output_bp, log_csv, var_name,
                          step_start, step_end, shape, tol, s, eb_mode,
                          compressor_type);
  } else {
    fprintf(stderr, "Unknown type: %s\n", dt.c_str()); MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
