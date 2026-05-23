#include <cuda_runtime.h>
#include <stdio.h>

struct TimerGPU {
    cudaEvent_t start, stop;
};

void iniciar_timer(TimerGPU *t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

void iniciar_timer_event(TimerGPU *t) {
    cudaEventRecord(t->start, 0);
}

float detener_timer_event(TimerGPU *t, const char *nombre) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    float ms;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    if (nombre != NULL) {
        printf("%s: %.3f ms\n", nombre, ms);
    }
    return ms;
}

void detener_timer(TimerGPU *t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}
