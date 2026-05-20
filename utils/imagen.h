#ifndef IMAGEN_H
#define IMAGEN_H

float* cargar_png_rgb(const char* nombre, int* H, int* W);
void guardar_png_gris(const char* nombre, float* datos, int H, int W);
void guardar_png_rgb(const char* nombre, float* datos, int B, int H, int W);

#endif