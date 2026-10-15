NVCC := nvcc
TARGET := gemm_baseline
SRC := src/main.cu
TEST_TMA := test_tma
TEST_TMA_SRC := tests/tma_copy_test.cu

NVCC_FLAGS := -O3 -std=c++17 -arch=sm_90a -lineinfo -I src -I benchmarks -I tests
LDLIBS := -lcublas -lcuda
HEADERS := $(wildcard src/common/*.cuh src/kernels/*.cuh benchmarks/*.cuh tests/*.cuh)

.PHONY: all run clean test-tma

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

test-tma: $(TEST_TMA)

$(TEST_TMA): $(TEST_TMA_SRC) $(wildcard src/common/*.cuh)
	$(NVCC) $(NVCC_FLAGS) $(TEST_TMA_SRC) -o $(TEST_TMA) -lcuda

clean:
	rm -f $(TARGET) $(TEST_TMA)
