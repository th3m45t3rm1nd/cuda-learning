#include <cstdio>
#include <stdio.h>
__global__ void hello() { printf("Hello from GPU!\n"); }

int main() {
  printf("Hello from CPU!\n");
  hello<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("Kernel execution error: %s\n", cudaGetErrorString(err));
  }
  return 0;
}
