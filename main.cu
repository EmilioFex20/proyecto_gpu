#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NOMINMAX
#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <string.h>
#include "utils/imagen.h"
#include "utils/timer.h"
#include "kernels/grises.h"
#include "kernels/bordes.h"
#include "kernels/normalizar.h"
#include "kernels/mse.h"

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

std::vector<std::string> obtener_archivos(const char *carpeta)
{
    std::vector<std::string> archivos;

    std::string patron = std::string(carpeta) + "\\*";
    WIN32_FIND_DATAA datos;
    HANDLE busqueda = FindFirstFileA(patron.c_str(), &datos);

    if (busqueda != INVALID_HANDLE_VALUE)
    {
        do
        {
            if ((datos.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
            {
                std::string nombre(datos.cFileName);
                std::string extension = nombre.length() > 4 ? nombre.substr(nombre.length() - 4) : "";
                std::transform(extension.begin(), extension.end(), extension.begin(),
                               [](unsigned char c)
                               { return (char)std::tolower(c); });

                if (extension == ".png" || extension == ".jpg")
                {
                    archivos.push_back(std::string(carpeta) + "/" + nombre);
                }
            }
        } while (FindNextFileA(busqueda, &datos));

        FindClose(busqueda);
    }
    else
    {
        fprintf(stderr, "No se puede abrir la carpeta %s\n", carpeta);
        exit(1);
    }

    std::sort(archivos.begin(), archivos.end());
    return archivos;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *carpeta = "imagenes";
    printf("Buscando imagenes en %s...\n", carpeta);
    std::vector<std::string> archivos = obtener_archivos(carpeta);
    int B = (int)archivos.size();

    if (B == 0)
    {
        fprintf(stderr, "No se encontraron imagenes .png o .jpg en %s\n", carpeta);
        return 1;
    }

    printf("Encontradas %d imagenes.\n", B);

    int H = 0, W = 0;
    printf("Cargando %s...\n", archivos[0].c_str());
    float *primera_img = cargar_png_rgb(archivos[0].c_str(), &H, &W);

    size_t pixeles = (size_t)H * (size_t)W;
    size_t bytes_rgb = 3 * pixeles * sizeof(float);
    size_t bytes_batch_rgb = (size_t)B * bytes_rgb;
    size_t bytes_batch_gris = (size_t)B * pixeles * sizeof(float);

    float *h_batch = (float *)malloc(bytes_batch_rgb);
    if (!h_batch)
    {
        fprintf(stderr, "No se pudo reservar memoria CPU para el batch.\n");
        free(primera_img);
        return 1;
    }

    memcpy(h_batch, primera_img, bytes_rgb);
    free(primera_img);

    for (int i = 1; i < B; i++)
    {
        int h_img = 0, w_img = 0;
        printf("Cargando %s...\n", archivos[i].c_str());
        float *img = cargar_png_rgb(archivos[i].c_str(), &h_img, &w_img);
        if (h_img != H || w_img != W)
        {
            fprintf(stderr, "Dimension incompatible en %s: %dx%d, se esperaba %dx%d\n",
                    archivos[i].c_str(), w_img, h_img, W, H);
            free(img);
            free(h_batch);
            return 1;
        }

        memcpy(h_batch + (size_t)i * 3 * pixeles, img, bytes_rgb);
        free(img);
    }

    printf("Batch listo: B=%d, W=%d, H=%d\n", B, W, H);

    if (!CreateDirectoryA("resultados", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
    {
        fprintf(stderr, "No se pudo crear la carpeta resultados (error Windows %lu)\n", GetLastError());
        free(h_batch);
        return 1;
    }

    printf("Reservando memoria GPU...\n");
    float *d_entrada, *d_grises, *d_bordes, *d_normalizado;
    float *d_maximos, *d_rmse, *d_referencia;
    CUDA_CHECK(cudaMalloc(&d_entrada, bytes_batch_rgb));
    CUDA_CHECK(cudaMalloc(&d_grises, bytes_batch_gris));
    CUDA_CHECK(cudaMalloc(&d_bordes, bytes_batch_gris));
    CUDA_CHECK(cudaMalloc(&d_normalizado, bytes_batch_gris));
    CUDA_CHECK(cudaMalloc(&d_maximos, B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rmse, B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_referencia, pixeles * sizeof(float)));

    printf("Copiando batch a GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_entrada, h_batch, bytes_batch_rgb, cudaMemcpyHostToDevice));

    dim3 bloque(16, 16);
    dim3 grid((W + bloque.x - 1) / bloque.x, (H + bloque.y - 1) / bloque.y);

    printf("Preparando referencia...\n");
    escala_grises<<<grid, bloque>>>(d_entrada, d_referencia, 1, H, W);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Ejecutando kernels...\n");
    TimerGPU timer;
    iniciar_timer(&timer);

    iniciar_timer_event(&timer);
    escala_grises<<<grid, bloque>>>(d_entrada, d_grises, B, H, W);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 1 - Grises");

    iniciar_timer_event(&timer);
    detectar_bordes<<<grid, bloque>>>(d_grises, d_bordes, B, H, W);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 2 - Bordes");

    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemset(d_maximos, 0, B * sizeof(float)));
    max_por_imagen<<<grid, bloque, bloque.x * bloque.y * sizeof(float)>>>(d_bordes, d_maximos, B, H, W);
    CUDA_KERNEL_CHECK();
    normalizar<<<grid, bloque>>>(d_bordes, d_maximos, d_normalizado, B, H, W);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 3 - Normalizacion");

    iniciar_timer_event(&timer);
    const int hilos_rmse = 256;
    calcular_rmse<<<B, hilos_rmse, hilos_rmse * sizeof(float)>>>(d_normalizado, d_referencia, d_rmse, B, H, W);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 4 - RMSE");

    detener_timer(&timer);

    printf("Copiando resultados a CPU...\n");
    float *h_grises = (float *)malloc(bytes_batch_gris);
    float *h_bordes = (float *)malloc(bytes_batch_gris);
    float *h_normalizado = (float *)malloc(bytes_batch_gris);
    float *h_rmse = (float *)malloc(B * sizeof(float));

    if (!h_grises || !h_bordes || !h_normalizado || !h_rmse)
    {
        fprintf(stderr, "No se pudo reservar memoria CPU para resultados.\n");
        return 1;
    }

    CUDA_CHECK(cudaMemcpy(h_grises, d_grises, bytes_batch_gris, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bordes, d_bordes, bytes_batch_gris, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normalizado, d_normalizado, bytes_batch_gris, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rmse, d_rmse, B * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Guardando resultados...\n");
    guardar_png_rgb("resultados/imagen_00_original.png", h_batch, 1, H, W);
    guardar_png_gris("resultados/imagen_00_grises.png", h_grises, H, W);
    guardar_png_gris("resultados/imagen_00_bordes.png", h_bordes, H, W);
    guardar_png_gris("resultados/imagen_00_normalizada.png", h_normalizado, H, W);

    FILE *f = fopen("resultados/rmse_por_imagen.txt", "w");
    if (!f)
    {
        fprintf(stderr, "No se pudo abrir resultados/rmse_por_imagen.txt para escritura\n");
        return 1;
    }

    for (int i = 0; i < B; i++)
        fprintf(f, "Imagen %02d: %f\n", i, h_rmse[i]);
    fclose(f);

    free(h_batch);
    free(h_grises);
    free(h_bordes);
    free(h_normalizado);
    free(h_rmse);
    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_grises));
    CUDA_CHECK(cudaFree(d_bordes));
    CUDA_CHECK(cudaFree(d_normalizado));
    CUDA_CHECK(cudaFree(d_maximos));
    CUDA_CHECK(cudaFree(d_rmse));
    CUDA_CHECK(cudaFree(d_referencia));

    printf("Pipeline terminado. Resultados guardados en resultados/.\n");
    return 0;
}
