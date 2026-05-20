### Kernel 1: Escala de grises

Este kernel convierte un batch de imágenes RGB (4D: B×3×H×W) a escala de grises (3D: B×H×W). 

- **Entrada:** tensor 4D de floats (RGB)  
- **Salida:** tensor 3D de floats (grises)  
- **Grid:** 2D con bloques de 16×16 hilos  
- **Fórmula usada:** Gris = 0.2989*R + 0.5870*G + 0.1140*B  
- **Notas:** Cada hilo procesa un píxel (fila, columna) de cada imagen, y un loop interno recorre las imágenes del batch. Se deja en la GPU la salida para pasar al siguiente kernel sin transferencias intermedias a CPU.

### Kernel 2: Detección de bordes (Sobel)

Este kernel aplica el filtro Sobel para detectar bordes en un batch de imágenes en escala de grises (3D: B×H×W).

- **Entrada:** tensor 3D de floats (grises)  
- **Salida:** tensor 3D de floats (bordes detectados)  
- **Grid:** 2D con bloques de 16×16 hilos  
- **Fórmula:** Magnitud = sqrt(Gx² + Gy²), donde Gx y Gy son convoluciones con los kernels Sobel estándar  
- **Notas:** Los píxeles de borde (primera y última fila/columna) se dejan en 0. Cada hilo procesa un píxel y un loop interno recorre todas las imágenes del batch. La salida permanece en GPU para el siguiente kernel.

### Kernel 3: Normalización

Este kernel normaliza cada imagen del batch al rango [0,1] en dos pasos:

1. **Reducción para máximo por imagen:**  
   - Encuentra el valor máximo de cada imagen usando memoria compartida (`__shared__`) y `atomicMax`.
2. **División de cada píxel:**  
   - Cada píxel se divide entre el máximo de su imagen para obtener valores entre 0 y 1.

- **Entrada:** tensor 3D de floats (bordes detectados)  
- **Salida:** tensor 3D de floats (normalizado)  
- **Grid:** 2D con bloques de 16×16 hilos  
- **Notas:** Cada imagen del batch mantiene su máximo independiente, y todo permanece en GPU para el siguiente kernel.

### Kernel 4: Cálculo de RMSE vs imagen de referencia

Este kernel compara cada imagen normalizada del batch contra una imagen de referencia única y devuelve un vector 1D con el RMSE por imagen.

- **Entrada:** tensor 3D de floats (normalizado), imagen de referencia 2D (H×W)  
- **Salida:** vector 1D de floats (RMSE por imagen)  
- **Grid:** 1D, reducción usando memoria compartida (`__shared__`)  
- **Fórmula:** 
  MSE[b] = promedio( (entrada[b][i] - referencia[i])² )  
  RMSE[b] = sqrt(MSE[b])  
- **Notas:** Cada hilo procesa un píxel, se realiza reducción en shared memory para obtener MSE y luego RMSE por imagen. La salida se mantiene en GPU para posterior transferencia a CPU.

## Main.cu — Orquestación del pipeline

Este archivo coordina la ejecución completa del pipeline:

1. Carga un batch de imágenes RGB desde `imagenes/` usando la función `cargar_png_rgb` de `utils/imagen.cu`.
2. Copia los datos a GPU (Host → Device).
3. Prepara una imagen de referencia en GPU (para RMSE).
4. Ejecuta los 4 kernels en orden:
   - **Kernel 1:** Escala de grises
   - **Kernel 2:** Bordes Sobel
   - **Kernel 3:** Normalización
   - **Kernel 4:** RMSE vs referencia
5. Mide el tiempo de ejecución de cada kernel y del pipeline completo usando `utils/timer.cu`.
6. Copia los resultados de GPU a CPU y guarda:
   - Imágenes: `imagen_00_original.png`, `imagen_00_grises.png`, `imagen_00_bordes.png`, `imagen_00_normalizada.png`
   - Archivo de RMSE: `rmse_por_imagen.txt`
7. Libera toda la memoria GPU y CPU utilizada.

**Notas:**  
- Todo el procesamiento se mantiene en GPU entre kernels.
- Las imágenes guardadas permiten verificar visualmente que cada kernel funciona correctamente.

### utils/timer.cu

Provee funciones para medir tiempos de ejecución en GPU usando `cudaEvent`. Se utiliza para calcular:

- Tiempo por kernel
- Tiempo total del pipeline

Funciones principales:

- `iniciar_timer` — inicializa eventos CUDA
- `iniciar_timer_event` — marca inicio del tiempo
- `detener_timer_event` — marca fin y calcula tiempo transcurrido
- `detener_timer` — libera eventos

### utils/imagen.cu

Funciones para cargar y guardar imágenes sin librerías externas complejas:

- `cargar_png_rgb(nombre, &H, &W)` — carga imagen RGB y retorna arreglo float [0,1]
- `guardar_png_gris(nombre, datos, H, W)` — guarda imagen en escala de grises
- `guardar_png_rgb(nombre, datos, B, H, W)` — guarda batch RGB

Usa las librerías de una sola cabecera `stb_image.h` y `stb_image_write.h`.