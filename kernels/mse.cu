/*
Entrada: tensor 3D (B, H, W) - batch normalizado

Referencia: tensor 2D (H, W) - imagen de referencia unica

Salida: tensor 1D (B,) - un RMSE por imagen del batch

Grid: 1 bloque por imagen, reduccion usando memoria compartida.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


// Kernel 4: calcula RMSE de cada imagen vs referencia.
__global__ void calcular_rmse(float *entrada, float *referencia, float *rmse, int B, int H, int W) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int b = blockIdx.x;
    int total = H * W;

    if (b >= B) return;

    float suma = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int idx = b * total + i;
        float diff = entrada[idx] - referencia[i];
        suma += diff * diff;
    }

    sdata[tid] = suma;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) rmse[b] = sqrtf(sdata[0] / total);
}
