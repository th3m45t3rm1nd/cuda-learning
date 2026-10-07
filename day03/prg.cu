#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
int main() {
  int n = 1'00'000'000;
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> c(n);

  std::iota(a.begin(), a.end(), 2);
  std::iota(b.begin(), b.end(), 3);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 8192;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << '\n';
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  float ms = 0;

  cudaEventElapsedTime(&ms, start, stop);

  printf("Kernel time: %f ms\n", ms);

  return 0;
}
