NVCC := nvcc
TARGET := gemm_baseline
SRC := h100_bf16_gemm_baseline.cu

NVCC_FLAGS := -O3 -std=c++17 -arch=sm_90a -lineinfo
LDLIBS := -lcublas -lcuda

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
