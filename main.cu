#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "kernels/grises.cu"
//#include "kernels/bordes.cu"
//#include "kernels/normalizar.cu"
//#include "kernels/mse.cu"
//#include "utils/imagen.cu"
//#include "utils/timer.cu"
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <errno.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#endif
#include "utils/imagen.h"
#include "utils/timer.h"
#include "kernels/grises.h"
#include "kernels/bordes.h"
#include "kernels/normalizar.h"
#include "kernels/mse.h"


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

std::vector<std::string> obtener_archivos(const char* carpeta) {
    std::vector<std::string> archivos;

#ifdef _WIN32
    std::string patron = std::string(carpeta) + "\\*";
    WIN32_FIND_DATAA datos;
    HANDLE handle = FindFirstFileA(patron.c_str(), &datos);
    if (handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "No se puede abrir la carpeta %s\n", carpeta);
        exit(1);
    }

    do {
        if (datos.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        std::string nombre(datos.cFileName);
        std::string lower = nombre;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return (char)std::tolower(ch); });

        if (lower.length() > 4 &&
            (lower.substr(lower.length() - 4) == ".png" ||
             lower.substr(lower.length() - 4) == ".jpg" ||
             (lower.length() > 5 && lower.substr(lower.length() - 5) == ".jpeg"))) {
            archivos.push_back(std::string(carpeta) + "/" + nombre);
        }
    } while (FindNextFileA(handle, &datos));

    FindClose(handle);
#else
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(carpeta)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string nombre(ent->d_name);
            std::string lower = nombre;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char ch) { return (char)std::tolower(ch); });
            if (lower.length() > 4 &&
                (lower.substr(lower.length()-4) == ".png" ||
                 lower.substr(lower.length()-4) == ".jpg" ||
                 (lower.length() > 5 && lower.substr(lower.length()-5) == ".jpeg"))) {
                archivos.push_back(std::string(carpeta) + "/" + nombre);
            }
        }
        closedir(dir);
    } else {
        fprintf(stderr, "No se puede abrir la carpeta %s\n", carpeta);
        exit(1);
    }
#endif

    // Ordenar para asegurar consistencia
    std::sort(archivos.begin(), archivos.end());
    return archivos;
}

void crear_directorio_si_no_existe(const char* ruta) {
#ifdef _WIN32
    if (_mkdir(ruta) != 0 && errno != EEXIST) {
#else
    if (mkdir(ruta, 0755) != 0 && errno != EEXIST) {
#endif
        fprintf(stderr, "No se pudo crear la carpeta %s\n", ruta);
        exit(1);
    }
}

int main() {
    // -------------------
    // Parámetros de prueba
    // -------------------
    printf("Inicio del pipeline\n");
    fflush(stdout);
    const char* carpeta = "imagenes";
    std::vector<std::string> archivos = obtener_archivos(carpeta);
    int B = archivos.size();

    if (B == 0) {
        fprintf(stderr, "No se encontraron imagenes .png o .jpg en la carpeta %s\n", carpeta);
        return 1;
    }

    int H = 0, W = 0;
    std::vector<float> h_batch;
    crear_directorio_si_no_existe("resultados");

    // Cargar imágenes al batch
    for (int i = 0; i < B; i++) {
        int imgH, imgW;
        float *img = cargar_png_rgb(archivos[i].c_str(), &imgH, &imgW);
        if (i == 0) {
            H = imgH;
            W = imgW;
            h_batch.resize((size_t)B * 3 * H * W);
        } else if (imgH != H || imgW != W) {
            fprintf(stderr, "La imagen %s mide %dx%d, pero se esperaba %dx%d\n",
                    archivos[i].c_str(), imgW, imgH, W, H);
            free(img);
            return 1;
        }
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    h_batch[i*3*H*W + c*H*W + y*W + x] = img[c*H*W + y*W + x];
        free(img);
    }
    printf("Batch copiado a GPU\n");
    // ------------------------------------
    // Reservar memoria GPU
    // ------------------------------------
    printf("Reservando memoria en GPU\n");
    float *d_entrada, *d_grises, *d_bordes, *d_normalizado;
    float *d_maximos, *d_rmse, *d_referencia;
    CUDA_CHECK(cudaMalloc(&d_entrada, B*3*H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grises, B*H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bordes, B*H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normalizado, B*H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maximos, B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rmse, B*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_referencia, H*W*sizeof(float)));

    printf("Memoria en GPU reservada\n");

    // Copiar batch a GPU
    CUDA_CHECK(cudaMemcpy(d_entrada, h_batch.data(), B*3*H*W*sizeof(float), cudaMemcpyHostToDevice));

    // Preparar referencia (ejemplo: primera imagen en gris)
    escala_grises<<<dim3((W+15)/16,(H+15)/16), dim3(16,16)>>>(d_entrada, d_referencia, 1, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Grid y block
    dim3 bloque(16,16);
    dim3 grid((W+15)/16, (H+15)/16);

    // ------------------------------------
    // Ejecutar kernels
    // ------------------------------------
    TimerGPU timer;
    iniciar_timer(&timer);

    // Kernel 1: Grises
    
    printf("Ejecutando kernel 1 - Grises\n");
    iniciar_timer_event(&timer);
    escala_grises<<<grid, bloque>>>(d_entrada, d_grises, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 1 - Grises");

    // Kernel 2: Bordes Sobel
    printf("Ejecutando kernel 2 - Bordes\n");
    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemset(d_bordes, 0, B*H*W*sizeof(float)));
    detectar_bordes<<<grid, bloque>>>(d_grises, d_bordes, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 2 - Bordes");

    // Kernel 3: Normalización
    printf("Ejecutando kernel 3 - Normalización\n");
    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemset(d_maximos, 0, B*sizeof(float)));
    max_por_imagen<<<grid, bloque, bloque.x*bloque.y*sizeof(float)>>>(d_bordes, d_maximos, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    normalizar<<<grid, bloque>>>(d_bordes, d_maximos, d_normalizado, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 3 - Normalización");

    // Kernel 4: RMSE
    printf("Ejecutando kernel 4 - RMSE\n");
    iniciar_timer_event(&timer);
    calcular_rmse<<<B, 256, 256*sizeof(float)>>>(d_normalizado, d_referencia, d_rmse, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    detener_timer_event(&timer, "Kernel 4 - RMSE");

    detener_timer(&timer);

    // ------------------------------------
    // Copiar resultados a CPU
    // ------------------------------------
    float *h_grises = (float*)malloc(B*H*W*sizeof(float));
    float *h_bordes = (float*)malloc(B*H*W*sizeof(float));
    float *h_normalizado = (float*)malloc(B*H*W*sizeof(float));
    float *h_rmse = (float*)malloc(B*sizeof(float));

    printf("Copiando resultados a CPU\n");

    CUDA_CHECK(cudaMemcpy(h_grises, d_grises, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bordes, d_bordes, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normalizado, d_normalizado, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rmse, d_rmse, B*sizeof(float), cudaMemcpyDeviceToHost));

    printf("Resultados copiados a CPU\n");

    // ------------------------------------
    // Guardar imágenes
    // ------------------------------------
    for (int i = 0; i < B; i++) {
        // Original RGB
        guardar_png_rgb(
            ("resultados/imagen_" + std::to_string(i) + "_original.png").c_str(),
            &h_batch[i*3*H*W], 1, H, W
        );

        // Grises
        guardar_png_gris(
            ("resultados/imagen_" + std::to_string(i) + "_grises.png").c_str(),
            &h_grises[i*H*W], H, W
        );

        // Bordes
        guardar_png_gris(
            ("resultados/imagen_" + std::to_string(i) + "_bordes.png").c_str(),
            &h_bordes[i*H*W], H, W
        );

        // Normalizado
        guardar_png_gris(
            ("resultados/imagen_" + std::to_string(i) + "_normalizada.png").c_str(),
            &h_normalizado[i*H*W], H, W
        );
    }

    printf("Imágenes guardadas\n");
    FILE *f = fopen("resultados/rmse_por_imagen.txt", "w");
    for (int i = 0; i < B; i++) fprintf(f, "Imagen %02d: %f\n", i, h_rmse[i]);
    fclose(f);

    // ------------------------------------
    // Liberar memoria
    // ------------------------------------
    printf("Liberando memoria\n");
    free(h_grises); free(h_bordes); free(h_normalizado); free(h_rmse);
    CUDA_CHECK(cudaFree(d_entrada)); CUDA_CHECK(cudaFree(d_grises)); CUDA_CHECK(cudaFree(d_bordes));
    CUDA_CHECK(cudaFree(d_normalizado)); CUDA_CHECK(cudaFree(d_maximos)); CUDA_CHECK(cudaFree(d_rmse));
    CUDA_CHECK(cudaFree(d_referencia));

    return 0;
}
