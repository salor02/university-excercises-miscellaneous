ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

ifndef EXERCISE
EXERCISE=exercise1.cu
endif

BUILD_DIR ?= ./build

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math `pkg-config --cflags --libs opencv4`

CXXFLAGS:=$(OPT) -I. $(EXT_CXXFLAGS)
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

SRCS:= utils.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE=$(EXERCISE:.cu=.exe)

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p
