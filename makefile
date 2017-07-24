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

make6:
	$(NCC) ./CUDAqueen2.cu -o ./bin/RCUDA2 $(NCCFLAGS) $(LIB)

make7:
	$(NCC) ./test/WorkSet2Test.cu -o ./bin/WSTEST $(NCCFLAGS) $(LIB)


run0-memcheck:
	cuda-memcheck ./bin/RCUDA

run1-memcheck:
	cuda-memcheck ./bin/RMEMTEST

run2-memcheck:
	cuda-memcheck ./bin/VTEST

run3-memcheck:
	cuda-memcheck ./bin/VCTEST

run4-memcheck:
	cuda-memcheck ./bin/QTEST

run5-memcheck:
	cuda-memcheck ./bin/QPTEST

run6-memcheck:
	cuda-memcheck ./bin/RCUDA2

run7-memcheck:
	cuda-memcheck ./bin/WSTEST


run0: 
	./bin/RCUDA

run1: 
	./bin/RMEMTEST

run2:
	./bin/VTEST

run3:
	./bin/VCTEST

run4:
	./bin/QTEST

run5:
	./bin/QPTEST

run6: 
	./bin/RCUDA2

run7: 
	./bin/WSTEST


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

clean6:
	rm ./bin/RCUDA2

clean7:
	rm ./bin/WSTEST


all: make0 make1 make2 make3 make4 make5 make6 make7
