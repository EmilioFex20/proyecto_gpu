/*
Entrada: tensor 3D (B, H, W) — batch con bordes  

Salida: tensor 3D (B, H, W) — cada imagen normalizada a [0, 1]  

Requiere: primero encontrar el valor máximo de cada imagen (reducción),

luego dividir cada píxel por ese máximo.

Este kernel se implementa en dos pasos:

Paso A: kernel de reducción → encuentra max por imagen (usa __shared__)
Paso B: kernel de división  → divide cada píxel entre el max de su imagen
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "timer.cu"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Paso A: kernel de reducción para obtener máximo por imagen
__global__ void max_por_imagen(float *entrada, float *maximos, int B, int H, int W) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    for (int b = 0; b < B; b++) {
        if (fila < H && col < W) {
            int idx = fila * W + col + b * H * W;
            sdata[tid] = entrada[idx];
        } else {
            sdata[tid] = 0.0f;
        }
        __syncthreads();

        // Reducción en shared memory
        for (int s = blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }

        // Guardar máximo del bloque
        if (tid == 0) atomicMax((int*)&maximos[b], __float_as_int(sdata[0]));
        __syncthreads();
    }
}

// Paso B: dividir cada píxel entre el máximo
__global__ void normalizar(float *entrada, float *maximos, float *salida, int B, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < H && col < W) {
        for (int b = 0; b < B; b++) {
            int idx = b * H * W + fila * W + col;
            salida[idx] = entrada[idx] / maximos[b];
        }
    }
}