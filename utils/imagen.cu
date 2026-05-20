#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

// Carga imagen RGB como float [0,1]
float* cargar_png_rgb(const char *nombre, int *H, int *W) {
    int C;
    unsigned char *data = stbi_load(nombre, W, H, &C, 3);
    if (!data) {
        fprintf(stderr, "Error cargando imagen %s\n", nombre);
        exit(1);
    }

    float *fdata = (float*)malloc(3 * (*H) * (*W) * sizeof(float));
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < *H; y++)
            for (int x = 0; x < *W; x++)
                fdata[c*(*H)*(*W) + y*(*W) + x] = data[(y*(*W)+x)*3 + c]/255.0f;

    stbi_image_free(data);
    return fdata;
}

// Guarda arreglo float [0,1] como imagen PNG en escala de grises
void guardar_png_gris(const char *nombre, float *datos, int H, int W) {
    unsigned char *out = (unsigned char*)malloc(H*W);
    for (int i = 0; i < H*W; i++)
        out[i] = (unsigned char)(fminf(fmaxf(datos[i],0.0f),1.0f)*255.0f);

    stbi_write_png(nombre, W, H, 1, out, W);
    free(out);
}

// Guarda arreglo float [0,1] como imagen RGB
void guardar_png_rgb(const char *nombre, float *datos, int B, int H, int W) {
    unsigned char *out = (unsigned char*)malloc(3*H*W);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                out[(y*W+x)*3+c] = (unsigned char)(fminf(fmaxf(datos[c*H*W + y*W + x],0.0f),1.0f)*255.0f);

    stbi_write_png(nombre, W, H, 3, out, W*3);
    free(out);
}