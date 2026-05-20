/*
Entrada: tensor 4D (B, 3, H, W) — batch de imágenes RGB  

Salida: tensor 3D (B, H, W) — batch de imágenes en gris  

Grid: 2D con dim3 bloque(16, 16), loop sobre imágenes del batch  

Fórmula: Gris = 0.2989*R + 0.5870*G + 0.1140*B
*/
#include <cuda_runtime.h>
#include <stdio.h>



// Kernel 1: convierte un batch de imágenes RGB a escala de grises
__global__ void escala_grises(float *entrada, float *salida, int B, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < H && col < W) {
        for (int b = 0; b < B; b++) {
            int base = b * 3 * H * W; // inicio de la imagen b en el batch
            int i = fila * W + col;
            salida[b * H * W + i] =
                0.2989f * entrada[base + 0 * H * W + i] +
                0.5870f * entrada[base + 1 * H * W + i] +
                0.1140f * entrada[base + 2 * H * W + i];
        }
    }
}