NVCC ?= nvcc
TARGET ?= pipeline

SOURCES := main.cu \
	kernels/grises.cu \
	kernels/bordes.cu \
	kernels/normalizar.cu \
	kernels/mse.cu \
	utils/imagen.cu \
	utils/timer.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) -std=c++14 -O2 -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o
