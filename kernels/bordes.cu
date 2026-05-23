/*
Entrada: tensor 3D (B, H, W) - batch en escala de grises

Salida: tensor 3D (B, H, W) - batch con bordes detectados

Grid: 2D con dim3 bloque(16, 16)

El filtro Sobel calcula el gradiente de intensidad en X y en Y.
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Kernel 2: detecta bordes con filtro Sobel en cada imagen del batch.
__global__ void detectar_bordes(float *entrada, float *salida, int B, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila >= H || col >= W) {
        return;
    }

    if (fila == 0 || fila == H - 1 || col == 0 || col == W - 1) {
        for (int b = 0; b < B; b++) {
            salida[b * H * W + fila * W + col] = 0.0f;
        }
        return;
    }

    for (int b = 0; b < B; b++) {
        int base = b * H * W;
        int i = fila * W + col;

        float gx =
            -entrada[base + (fila - 1) * W + (col - 1)] + entrada[base + (fila - 1) * W + (col + 1)] +
            -2.0f * entrada[base + fila * W + (col - 1)] + 2.0f * entrada[base + fila * W + (col + 1)] +
            -entrada[base + (fila + 1) * W + (col - 1)] + entrada[base + (fila + 1) * W + (col + 1)];

        float gy =
            -entrada[base + (fila - 1) * W + (col - 1)] - 2.0f * entrada[base + (fila - 1) * W + col] - entrada[base + (fila - 1) * W + (col + 1)] +
            entrada[base + (fila + 1) * W + (col - 1)] + 2.0f * entrada[base + (fila + 1) * W + col] + entrada[base + (fila + 1) * W + (col + 1)];

        salida[base + i] = sqrtf(gx * gx + gy * gy);
    }
}
