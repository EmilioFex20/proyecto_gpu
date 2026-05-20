/*
Entrada: tensor 3D (B, H, W) — batch en escala de grises  

Salida: tensor 3D (B, H, W) — batch con bordes detectados  

Grid: 2D con dim3 bloque(16, 16)

El filtro Sobel calcula el gradiente de intensidad en X y en Y:
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


// Kernel 2: Detecta bordes con filtro Sobel en cada imagen del batch
__global__ void detectar_bordes(float *entrada, float *salida, int B, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila <= 0 || fila >= H-1 || col <= 0 || col >= W-1)
        return; // píxeles de borde se dejan en 0

    for (int b = 0; b < B; b++) {
        int base = b * H * W;
        int i = fila * W + col;

        // Filtro Sobel en X
        float Gx =
            -entrada[base + (fila-1)*W + (col-1)] + entrada[base + (fila-1)*W + (col+1)] +
            -2*entrada[base + fila*W + (col-1)] + 2*entrada[base + fila*W + (col+1)] +
            -entrada[base + (fila+1)*W + (col-1)] + entrada[base + (fila+1)*W + (col+1)];

        // Filtro Sobel en Y
        float Gy =
            -entrada[base + (fila-1)*W + (col-1)] - 2*entrada[base + (fila-1)*W + col] - entrada[base + (fila-1)*W + (col+1)] +
            entrada[base + (fila+1)*W + (col-1)] + 2*entrada[base + (fila+1)*W + col] + entrada[base + (fila+1)*W + (col+1)];

        salida[base + i] = sqrtf(Gx*Gx + Gy*Gy);
    }
}