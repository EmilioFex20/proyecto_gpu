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
#include <chrono>
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

double ejecutar_pipeline_cpu_equivalente(const std::vector<float>& entrada,
                                         int B, int H, int W,
                                         float *checksum_rmse) {
    const int total = H * W;
    std::vector<float> referencia(total);
    std::vector<float> grises((size_t)B * total);
    std::vector<float> bordes((size_t)B * total, 0.0f);
    std::vector<float> maximos(B, 0.0f);
    std::vector<float> normalizado((size_t)B * total);
    std::vector<float> rmse(B);

    auto inicio = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int i = y * W + x;
            referencia[i] =
                0.2989f * entrada[0 * H * W + i] +
                0.5870f * entrada[1 * H * W + i] +
                0.1140f * entrada[2 * H * W + i];
        }
    }

    for (int b = 0; b < B; b++) {
        int base_rgb = b * 3 * H * W;
        int base = b * total;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int i = y * W + x;
                grises[base + i] =
                    0.2989f * entrada[base_rgb + 0 * H * W + i] +
                    0.5870f * entrada[base_rgb + 1 * H * W + i] +
                    0.1140f * entrada[base_rgb + 2 * H * W + i];
            }
        }
    }

    for (int b = 0; b < B; b++) {
        int base = b * total;
        for (int y = 1; y < H - 1; y++) {
            for (int x = 1; x < W - 1; x++) {
                int i = y * W + x;
                float Gx =
                    -grises[base + (y - 1) * W + (x - 1)] + grises[base + (y - 1) * W + (x + 1)] +
                    -2.0f * grises[base + y * W + (x - 1)] + 2.0f * grises[base + y * W + (x + 1)] +
                    -grises[base + (y + 1) * W + (x - 1)] + grises[base + (y + 1) * W + (x + 1)];

                float Gy =
                    -grises[base + (y - 1) * W + (x - 1)] - 2.0f * grises[base + (y - 1) * W + x] - grises[base + (y - 1) * W + (x + 1)] +
                    grises[base + (y + 1) * W + (x - 1)] + 2.0f * grises[base + (y + 1) * W + x] + grises[base + (y + 1) * W + (x + 1)];

                bordes[base + i] = sqrtf(Gx * Gx + Gy * Gy);
            }
        }
    }

    for (int b = 0; b < B; b++) {
        int base = b * total;
        float maximo = 0.0f;
        for (int i = 0; i < total; i++) {
            maximo = fmaxf(maximo, bordes[base + i]);
        }
        maximos[b] = maximo;

        for (int i = 0; i < total; i++) {
            normalizado[base + i] = maximo > 0.0f ? bordes[base + i] / maximo : 0.0f;
        }
    }

    for (int b = 0; b < B; b++) {
        int base = b * total;
        double suma = 0.0;
        for (int i = 0; i < total; i++) {
            double diff = (double)normalizado[base + i] - (double)referencia[i];
            suma += diff * diff;
        }
        rmse[b] = sqrtf((float)(suma / total));
    }

    auto fin = std::chrono::high_resolution_clock::now();

    float checksum = 0.0f;
    for (int b = 0; b < B; b++) {
        checksum += rmse[b];
    }
    *checksum_rmse = checksum;

    std::chrono::duration<double, std::milli> duracion = fin - inicio;
    return duracion.count();
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
    printf("Batch cargado en CPU\n");

    float checksum_cpu = 0.0f;
    printf("Ejecutando implementación CPU equivalente\n");
    double tiempo_cpu_ms = ejecutar_pipeline_cpu_equivalente(h_batch, B, H, W, &checksum_cpu);
    printf("Tiempo CPU equivalente: %.3f ms\n", tiempo_cpu_ms);
    if (checksum_cpu < 0.0f) {
        printf("Checksum CPU: %.6f\n", checksum_cpu);
    }
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

    TimerGPU timer;
    iniciar_timer(&timer);

    float h2d_ms = 0.0f;
    float d2h_ms = 0.0f;
    float referencia_ms = 0.0f;
    float kernel1_ms = 0.0f;
    float kernel2_ms = 0.0f;
    float kernel3_ms = 0.0f;
    float kernel4_ms = 0.0f;
    float tiempo_pipeline_ms = 0.0f;

    // Copiar batch a GPU
    printf("Copiando batch a GPU\n");
    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemcpy(d_entrada, h_batch.data(), B*3*H*W*sizeof(float), cudaMemcpyHostToDevice));
    h2d_ms = detener_timer_event(&timer, "Transferencia H→D");
    printf("Batch copiado a GPU\n");

    // Preparar referencia (ejemplo: primera imagen en gris)
    iniciar_timer_event(&timer);
    escala_grises<<<dim3((W+15)/16,(H+15)/16), dim3(16,16)>>>(d_entrada, d_referencia, 1, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    referencia_ms = detener_timer_event(&timer, NULL);

    // Grid y block
    dim3 bloque(16,16);
    dim3 grid((W+15)/16, (H+15)/16);

    // ------------------------------------
    // Ejecutar kernels
    // ------------------------------------
    // Kernel 1: Grises
    
    printf("Ejecutando kernel 1 - Grises\n");
    iniciar_timer_event(&timer);
    escala_grises<<<grid, bloque>>>(d_entrada, d_grises, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel1_ms = detener_timer_event(&timer, "Kernel 1 - Grises");

    // Kernel 2: Bordes Sobel
    printf("Ejecutando kernel 2 - Bordes\n");
    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemset(d_bordes, 0, B*H*W*sizeof(float)));
    detectar_bordes<<<grid, bloque>>>(d_grises, d_bordes, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel2_ms = detener_timer_event(&timer, "Kernel 2 - Bordes");

    // Kernel 3: Normalización
    printf("Ejecutando kernel 3 - Normalización\n");
    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemset(d_maximos, 0, B*sizeof(float)));
    max_por_imagen<<<grid, bloque, bloque.x*bloque.y*sizeof(float)>>>(d_bordes, d_maximos, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    normalizar<<<grid, bloque>>>(d_bordes, d_maximos, d_normalizado, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel3_ms = detener_timer_event(&timer, "Kernel 3 - Normalización");

    // Kernel 4: RMSE
    printf("Ejecutando kernel 4 - RMSE\n");
    iniciar_timer_event(&timer);
    calcular_rmse<<<B, 256, 256*sizeof(float)>>>(d_normalizado, d_referencia, d_rmse, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel4_ms = detener_timer_event(&timer, "Kernel 4 - RMSE");

    // ------------------------------------
    // Copiar resultados a CPU
    // ------------------------------------
    float *h_grises = (float*)malloc(B*H*W*sizeof(float));
    float *h_bordes = (float*)malloc(B*H*W*sizeof(float));
    float *h_normalizado = (float*)malloc(B*H*W*sizeof(float));
    float *h_rmse = (float*)malloc(B*sizeof(float));

    printf("Copiando resultados a CPU\n");

    iniciar_timer_event(&timer);
    CUDA_CHECK(cudaMemcpy(h_grises, d_grises, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bordes, d_bordes, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normalizado, d_normalizado, B*H*W*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rmse, d_rmse, B*sizeof(float), cudaMemcpyDeviceToHost));
    d2h_ms = detener_timer_event(&timer, "Transferencia D→H");
    tiempo_pipeline_ms = kernel1_ms + kernel2_ms + kernel3_ms + kernel4_ms;

    printf("Resultados copiados a CPU\n");
    printf("Tiempo total del pipeline: %.3f ms\n", tiempo_pipeline_ms);
    printf("Tiempo de las transferencias H→D y D→H: %.3f ms (H→D: %.3f ms, D→H: %.3f ms)\n",
           h2d_ms + d2h_ms, h2d_ms, d2h_ms);
    printf("Speedup del pipeline completo vs implementación CPU equivalente: %.2fx\n",
           tiempo_pipeline_ms > 0.0f ? tiempo_cpu_ms / tiempo_pipeline_ms : 0.0);
    printf("Tiempo extra preparando referencia GPU: %.3f ms\n", referencia_ms);

    detener_timer(&timer);

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
