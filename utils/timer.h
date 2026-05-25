// utils/timer.h
#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include <stdio.h>

struct TimerGPU {
    cudaEvent_t start, stop;
};

void iniciar_timer(TimerGPU* t);
void iniciar_timer_event(TimerGPU* t);
float detener_timer_event(TimerGPU* t, const char* nombre);
void detener_timer(TimerGPU* t);

#endif
