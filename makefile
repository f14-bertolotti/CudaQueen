CC		= g++
NCC		= nvcc
CFLAGS	= -Wall -Wextra -std=c++11
NCCFLAGS	= --std=c++11 -G -arch=sm_35 -rdc=true
LIB			= -L/usr/lib/x86_64-linux-gnu/


make:
	$(NCC) ./CUDAqueen.cu -o ./bin/RCUDA $(NCCFLAGS) $(LIB)


run-memcheck:
	cuda-memcheck ./bin/RCUDA

run: 
	./bin/RCUDA

clean:
	rm ./bin/RCUDA

all: make
