CC		= g++
NCC		= nvcc
CFLAGS	= -Wall -Wextra -std=c++11
NCCFLAGS	= --std=c++11 -g -G -arch=sm_35 -rdc=true
LIB			= -L/usr/lib/x86_64-linux-gnu/


make1:
	$(NCC) ./test/MemoryManagementTest.cu -o ./bin/RMEMTEST $(NCCFLAGS) $(LIB)

make2:
	$(NCC) ./test/VariableTest.cu -o ./bin/VTEST $(NCCFLAGS) $(LIB)

make3:
	$(NCC) ./test/VariableCollectionTest.cu -o ./bin/VCTEST $(NCCFLAGS) $(LIB)

make4:
	$(NCC) ./test/TripleQueueTest.cu -o ./bin/QTEST $(NCCFLAGS) $(LIB)

make5:
	$(NCC) ./test/QueenPropagationTest.cu -o ./bin/QPTEST $(NCCFLAGS) $(LIB)

make6:
	$(NCC) ./CUDAqueen.cu -o ./bin/RCUDA $(NCCFLAGS) $(LIB)

make7:
	$(NCC) ./test/WorkSetTest.cu -o ./bin/WSTEST $(NCCFLAGS) $(LIB)




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
	cuda-memcheck ./bin/RCUDA

run7-memcheck:
	cuda-memcheck ./bin/WSTEST



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
	./bin/RCUDA

run7: 
	./bin/WSTEST



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


all: make1 make2 make3 make4 make5 make6 make7
