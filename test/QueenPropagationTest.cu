#include <stdio.h>
#include "../QueenPropagation/QueenPropagation.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"

__device__ DeviceVariableCollection deviceVariableCollection1;
__device__ DeviceVariableCollection deviceVariableCollection2;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceQueenConstraints deviceQueenConstraints;

__global__ void init1(int*,DeviceVariable*,int,int*,Triple*);
__global__ void init2(int*,DeviceVariable*,int,int*,Triple*);
__global__ void testSequential1();
__global__ void testParallel1();
__global__ void testParallel2();

int main(){

	HostVariableCollection hostVariableCollection1(6);
	HostVariableCollection hostVariableCollection2(5);
	init1<<<1,1>>>(hostVariableCollection1.dMem, 
				  hostVariableCollection1.dMemVariables,
				  hostVariableCollection1.nQueen,
				  hostVariableCollection1.dMemlastValues,
				  hostVariableCollection1.hostQueue.dMem);
	init2<<<1,1>>>(hostVariableCollection2.dMem, 
				  hostVariableCollection2.dMemVariables,
				  hostVariableCollection2.nQueen,
				  hostVariableCollection2.dMemlastValues,
				  hostVariableCollection2.hostQueue.dMem);
	cudaDeviceSynchronize();
	testSequential1<<<1,1>>>();
	testParallel1<<<1,1>>>();
	testParallel2<<<1,1>>>();

	cudaDeviceSynchronize();
	return 0;
}

__global__ void init1(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection1.init(deviceVariable,q,dMem,lv,nQueen);
}

__global__ void init2(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection2.init(deviceVariable,q,dMem,lv,nQueen);
}

__global__ void testSequential1(){

	printf("/////////SEQUENTIAL TEST/////////\n");

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,0));
	deviceVariableCollection1.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,1));
	deviceVariableCollection1.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceVariableCollection1.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

		deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceVariableCollection1.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceVariableCollection1.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceVariableCollection1.print();
}

__global__ void testParallel1(){

	printf("/////////PARALLEL TEST 6 QUEEN/////////\n");

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,0));
	deviceVariableCollection1.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,1));
	deviceVariableCollection1.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

	deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection1);
	deviceVariableCollection1.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,3,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,3));
	deviceVariableCollection1.print();

	if(deviceQueenConstraints.parallelConstraints(deviceVariableCollection1))printf("ok\n");
	else printf("not ok\n");
}

__global__ void testParallel2(){

	printf("/////////PARALLEL TEST 5 QUEEN/////////\n");

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection2,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection2,0));
	deviceVariableCollection2.print();

	int val = deviceQueenPropagation.nextAssign(deviceVariableCollection2,1);
	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection2,1,val);	
	deviceVariableCollection2.print();

}