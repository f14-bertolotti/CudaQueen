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

make3:
	$(NCC) ./test/VariableCollection2Test.cu -o ./bin/VCTEST $(NCCFLAGS) $(LIB)

make4:
	$(NCC) ./test/TripleQueue2Test.cu -o ./bin/QTEST $(NCCFLAGS) $(LIB)

make5:
	$(NCC) ./test/QueenPropagation2Test.cu -o ./bin/QPTEST $(NCCFLAGS) $(LIB)


run0-memcheck:
	make make0 && cuda-memcheck ./bin/RCUDA

run1-memcheck:
	make make1 && cuda-memcheck ./bin/RMEMTEST

run2-memcheck:
	make make2 && cuda-memcheck ./bin/VTEST

run3-memcheck:
	make make3 && cuda-memcheck ./bin/VCTEST

run4-memcheck:
	make make4 && cuda-memcheck ./bin/QTEST

run5-memcheck:
	make make5 && cuda-memcheck ./bin/QPTEST


run0: 
	make make0 && ./bin/RCUDA

run1: 
	make make1 && ./bin/RMEMTEST

run2:
	make make2 && ./bin/VTEST

run3:
	make make3 && ./bin/VCTEST

run4:
	make make4 && ./bin/QTEST

run5:
	make make5 && ./bin/QPTEST


clean0:
	rm ./bin/RCUDA

clean1:
	rm ./bin/RMEMTEST

clean2:
	rm ./bin/VTEST

clean3:
	rm ./bin/VCTEST

clean4:
	rm ./bin/QTEST

clean5:
	rm ./bin/QPTEST


all: make0 make1 make2 make3 make4 make5
