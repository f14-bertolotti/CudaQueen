CC		= g++
NCC		= nvcc
CFLAGS	= -Wall -Wextra -std=c++11
NCCFLAGS	= --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB			= -L/usr/lib/x86_64-linux-gnu/

make0:
	$(NCC) ./CUDAqueen.cu -o ./bin/RCUDA $(NCCFLAGS) $(LIB)

run-memcheck:
	make make0 && cuda-memcheck ./bin/RCUDA

run: 
	make make0 && ./bin/RCUDA

all: make0
