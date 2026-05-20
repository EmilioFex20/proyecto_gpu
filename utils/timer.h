#ifndef TIMER_H
#define TIMER_H

struct TimerGPU;
void iniciar_timer(TimerGPU* t);
void iniciar_timer_event(TimerGPU* t);
void detener_timer_event(TimerGPU* t, const char* nombre);
void detener_timer(TimerGPU* t);

#endif