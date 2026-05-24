#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK_TIMER(ans) { timerAssert((ans), __FILE__, __LINE__); }
inline void timerAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct TimerGPU {
    cudaEvent_t start, stop;
};

void iniciar_timer(TimerGPU *t) {
    CUDA_CHECK_TIMER(cudaEventCreate(&t->start));
    CUDA_CHECK_TIMER(cudaEventCreate(&t->stop));
}

void iniciar_timer_event(TimerGPU *t) {
    CUDA_CHECK_TIMER(cudaEventRecord(t->start, 0));
}

float detener_timer_event(TimerGPU *t, const char *nombre) {
    CUDA_CHECK_TIMER(cudaEventRecord(t->stop, 0));
    CUDA_CHECK_TIMER(cudaEventSynchronize(t->stop));
    float ms;
    CUDA_CHECK_TIMER(cudaEventElapsedTime(&ms, t->start, t->stop));
    if (nombre != NULL) {
        printf("%s: %.3f ms\n", nombre, ms);
    }
    return ms;
}

void detener_timer(TimerGPU *t) {
    CUDA_CHECK_TIMER(cudaEventDestroy(t->start));
    CUDA_CHECK_TIMER(cudaEventDestroy(t->stop));
}
