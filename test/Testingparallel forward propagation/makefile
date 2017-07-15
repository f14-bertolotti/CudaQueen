CC              = g++
NCC             = nvcc
CFLAGS  = -Wall -Wextra -std=c++11
NCCFLAGS        = --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB                     = -L/usr/lib/x86_64-linux-gnu/

make0:
	$(NCC) ./parallelForwardPropagation.cu -o ./parallelForwardPropagation $(NCCFLAGS) $(LIB)

run-memcheck: make0
	cuda-memcheck parallelForwardPropagation

run: make0
	./parallelForwardPropagation

all: make0
