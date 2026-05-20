#ifndef NORMALIZAR_H
#define NORMALIZAR_H

__global__ void max_por_imagen(float *entrada, float *maximos, int B, int H, int W);
__global__ void normalizar(float *entrada, float *maximos, float *salida, int B, int H, int W);

#endif