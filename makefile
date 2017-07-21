CC		= g++
NCC		= nvcc
CFLAGS	= -Wall -Wextra -std=c++11
NCCFLAGS	= --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB			= -L/usr/lib/x86_64-linux-gnu/

make0:
	$(NCC) ./CUDAqueen.cu -o ./bin/RCUDA $(NCCFLAGS) $(LIB)

make1:
	$(NCC) ./test/MemoryManagementTest.cu -o ./bin/RMEMTEST $(NCCFLAGS) $(LIB)

make2:
	$(NCC) ./test/Variable2Test.cu -o ./bin/VTEST $(NCCFLAGS) $(LIB)


run0-memcheck:
	make make0 && cuda-memcheck ./bin/RCUDA

run1-memcheck:
	make make1 && cuda-memcheck ./bin/RMEMTEST

run2-memcheck:
	make make2 && cuda-memcheck ./bin/VTEST


run0: 
	make make0 && ./bin/RCUDA

run1: 
	make make0 && ./bin/RMEMTEST

run2:
	cuda-memcheck ./bin/VTEST


clean0:
	rm ./bin/RCUDA

clean1:
	rm ./bin/RMEMTEST

clean2:
	rm ./bin/VTEST

all: make0
