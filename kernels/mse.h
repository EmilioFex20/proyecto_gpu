#ifndef MSE_H
#define MSE_H

__global__ void calcular_rmse(float *entrada, float *referencia, float *rmse, int B, int H, int W);

#endif