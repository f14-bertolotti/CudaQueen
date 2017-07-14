CC              = g++
NCC             = nvcc
CFLAGS  = -Wall -Wextra -std=c++11
NCCFLAGS        = --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB                     = -L/usr/lib/x86_64-linux-gnu/

make0:
	$(NCC) ./testPQueue.cu -o ./testPQueue $(NCCFLAGS) $(LIB)

run-memcheck:
	make make0 && cuda-memcheck testPQueue

run:
	make make0 && ./testPQueue

all: make0


