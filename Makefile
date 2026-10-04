NVCC := nvcc
TARGET := gemm_baseline
SRC := src/main.cu

NVCC_FLAGS := -O3 -std=c++17 -arch=sm_90a -lineinfo -I src -I benchmarks -I tests
LDLIBS := -lcublas -lcuda
HEADERS := $(wildcard src/common/*.cuh src/kernels/*.cuh benchmarks/*.cuh tests/*.cuh)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
