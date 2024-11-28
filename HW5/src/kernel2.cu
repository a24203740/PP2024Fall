#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_WIDTH 16

/**************************************************************************************/
/* CUDA MEMCHECK */
/* ref: */
/* https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api */
/**************************************************************************************/
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = false) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      getchar();
      exit(code);
    }
  }
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY,
                             size_t pitch, int maxIterations, int *d_img) {
  // To avoid error caused by the floating number, use the following pseudo code
  int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
  int tIDy = blockIdx.y * blockDim.y + threadIdx.y;
  float x = lowerX + tIDx * stepX;
  float y = lowerY + tIDy * stepY;

  float z_x = x, z_y = y;
  int i;
  for (i = 0; i < maxIterations; ++i) {
    if (z_x * z_x + z_y * z_y > 4.f)
      break;
    float new_x = z_x * z_x - z_y * z_y;
    float new_y = 2.f * z_x * z_y;
    z_x = x + new_x;
    z_y = y + new_y;
  }
  int* target = (int*)((char*)d_img + tIDy * pitch) + tIDx;
  *target = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;
  size_t imgSize = resX * resY * sizeof(int);
  int *h_img;
  int *d_img;
  size_t pitch = 0;
  gpuErrchk(cudaHostAlloc(&h_img, imgSize, cudaHostAllocDefault));
  gpuErrchk(cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY));
  gpuErrchk(cudaMemset(d_img, 0, imgSize));

  dim3 dimGrid(resX / TILE_WIDTH, resY / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, pitch, maxIterations, d_img);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_img));

  memcpy(img, h_img, imgSize);

  gpuErrchk(cudaFreeHost(h_img));
}
