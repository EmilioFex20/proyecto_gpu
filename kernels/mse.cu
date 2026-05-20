/*
Entrada: tensor 3D (B, H, W) — batch normalizado  

Referencia: tensor 2D (H, W) — imagen de referencia única  

Salida: tensor 1D (B,) — un RMSE por imagen del batch  

Grid: 1D con reducción usando __shared__

Para cada imagen b del batch:
  MSE[b] = promedio( (resultado[b][i] - referencia[i])² )
  RMSE[b] = sqrt(MSE[b])
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel 4: calcula RMSE de cada imagen vs referencia
__global__ void calcular_rmse(float *entrada, float *referencia, float *rmse, int B, int H, int W) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        // Cada hilo procesa un pixel del flatten H*W
        if (idx_global < H*W) {
            int idx = b * H * W + idx_global;
            float diff = entrada[idx] - referencia[idx_global];
            sdata[tid] = diff * diff;
        } else {
            sdata[tid] = 0.0f;
        }
        __syncthreads();

        // Reducción en shared memory para MSE
        for (int s = blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0) rmse[b] = sqrtf(sdata[0] / (H*W));
        __syncthreads();
    }
}