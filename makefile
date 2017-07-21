CC		= g++
NCC		= nvcc
CFLAGS	= -Wall -Wextra -std=c++11
NCCFLAGS	= --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB			= -L/usr/lib/x86_64-linux-gnu/

make0:
	$(NCC) ./CUDAqueen.cu -o ./bin/RCUDA $(NCCFLAGS) $(LIB)

make1:
	$(NCC) ./MemoryManagement/MemoryManagementTest.cu -o ./bin/RMEMTEST $(NCCFLAGS) $(LIB)

run0-memcheck:
	make make0 && cuda-memcheck ./bin/RCUDA

run1-memcheck:
	make make1 && cuda-memcheck ./bin/RMEMTEST

run0: 
	make make0 && ./bin/RCUDA

run1: 
	make make0 && ./bin/RMEMTEST

clean0:
	rm ./bin/RCUDA

clean1:
	rm ./bin/RMEMTEST

all: make0
