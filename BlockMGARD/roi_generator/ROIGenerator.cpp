/*
 * Simple ROI Tolerance Map Generator
 * Usage: ./generate_roi_map -o output.bin -dim nx ny nz -bg <bg_tol> -roi <roi_tol> x0 x1 y0 y1 z0 z1
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

void print_usage() {
  std::cout << "Usage: ./generate_roi_map -o <output> -dim <nx> <ny> <nz> -bg <bg_tol> [ROI options]\n\n"
            << "Required:\n"
            << "  -o <file>           Output binary file\n"
            << "  -dim <nx> <ny> <nz> Data dimensions (voxels)\n"
            << "  -bg <tol>           Background tolerance\n\n"
            << "Optional ROI (can specify multiple):\n"
            << "  -roi <tol> <x0> <x1> <y0> <y1> <z0> <z1>\n"
            << "       Voxel coordinates for ROI region\n\n"
            << "Examples:\n"
            << "  # Uniform tolerance\n"
            << "  ./generate_roi_map -o roi.bin -dim 512 512 512 -bg 1e-2\n\n"
            << "  # One ROI region\n"
            << "  ./generate_roi_map -o roi.bin -dim 512 512 512 -bg 1e-2 \\\n"
            << "    -roi 1e-5 200 300 200 300 200 300\n\n"
            << "  # Multiple ROI regions\n"
            << "  ./generate_roi_map -o roi.bin -dim 512 512 512 -bg 1e-2 \\\n"
            << "    -roi 1e-5 200 300 200 300 200 300 \\\n"
            << "    -roi 1e-4 100 200 150 250 200 300\n";
}

struct ROI {
  double tolerance;
  size_t x0, x1, y0, y1, z0, z1;  // voxel coordinates
};

int main(int argc, char* argv[]) {
  if (argc < 7) {
    print_usage();
    return 1;
  }

  std::string output_file;
  size_t nx = 0, ny = 0, nz = 0;
  double bg_tol = 0;
  std::vector<ROI> rois;

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    
    if (arg == "-o" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "-dim" && i + 3 < argc) {
      nx = atoi(argv[++i]);
      ny = atoi(argv[++i]);
      nz = atoi(argv[++i]);
    } else if (arg == "-bg" && i + 1 < argc) {
      bg_tol = atof(argv[++i]);
    } else if (arg == "-roi" && i + 7 < argc) {
      ROI roi;
      roi.tolerance = atof(argv[++i]);
      roi.x0 = atoi(argv[++i]);
      roi.x1 = atoi(argv[++i]);
      roi.y0 = atoi(argv[++i]);
      roi.y1 = atoi(argv[++i]);
      roi.z0 = atoi(argv[++i]);
      roi.z1 = atoi(argv[++i]);
      rois.push_back(roi);
    }
  }

  // Validate
  if (output_file.empty() || nx == 0 || ny == 0 || nz == 0) {
    std::cerr << "Error: Missing required arguments\n";
    print_usage();
    return 1;
  }

  // Calculate blocks (8x8x8)
  const size_t BLOCK_SIZE = 8;
  size_t nbx = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t nby = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t nbz = (nz + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t total = nbx * nby * nbz;

  std::cout << "Data shape: " << nx << " x " << ny << " x " << nz << "\n";
  std::cout << "Block shape: " << nbx << " x " << nby << " x " << nbz 
            << " = " << total << " blocks\n";
  std::cout << "Background tolerance: " << bg_tol << "\n";
  std::cout << "Number of ROIs: " << rois.size() << "\n";

  // Create tolerance map
  std::vector<double> tol_map(total, bg_tol);

  // Apply ROIs
  for (size_t r = 0; r < rois.size(); r++) {
    // Convert voxel to block coordinates
    size_t bx0 = rois[r].x0 / BLOCK_SIZE;
    size_t bx1 = (rois[r].x1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t by0 = rois[r].y0 / BLOCK_SIZE;
    size_t by1 = (rois[r].y1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bz0 = rois[r].z0 / BLOCK_SIZE;
    size_t bz1 = (rois[r].z1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Clamp to valid range
    if (bx1 > nbx) bx1 = nbx;
    if (by1 > nby) by1 = nby;
    if (bz1 > nbz) bz1 = nbz;

    std::cout << "ROI " << r << ": tolerance=" << rois[r].tolerance 
              << ", blocks=[" << bx0 << ":" << bx1 << ", "
              << by0 << ":" << by1 << ", " << bz0 << ":" << bz1 << "]\n";

    // Set tolerance for this ROI
    for (size_t bx = bx0; bx < bx1; bx++) {
      for (size_t by = by0; by < by1; by++) {
        for (size_t bz = bz0; bz < bz1; bz++) {
          size_t idx = bx * nby * nbz + by * nbz + bz;
          tol_map[idx] = rois[r].tolerance;
        }
      }
    }
  }

  // Write to file
  std::ofstream file(output_file.c_str(), std::ios::binary);
  if (!file) {
    std::cerr << "Error: Cannot open output file\n";
    return 1;
  }

  file.write(reinterpret_cast<const char*>(&tol_map[0]), total * sizeof(double));
  file.close();

  std::cout << "\nSuccess! Wrote " << total << " values (" 
            << total * sizeof(double) << " bytes) to " << output_file << "\n";

  return 0;
}