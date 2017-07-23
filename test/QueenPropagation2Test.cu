#include <stdio.h>
#include "../QueenPropagation/QueenPropagation2.cu"
#include "../VariableCollection/VariableCollection2.cu"

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceQueenPropagation deviceQueenPropagation;

__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void testSequential();
__global__ void testParallel();

int main(){

	HostVariableCollection hostVariableCollection(6);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.dMemVariables,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);
	testSequential<<<1,1>>>();
	testParallel<<<1,1>>>();

	cudaDeviceSynchronize();
	return 0;
}

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

__global__ void testSequential(){

	printf("/////////SEQUENTIAL TEST/////////\n");

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,0));
	deviceVariableCollection.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,1));
	deviceVariableCollection.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,2));
	deviceVariableCollection.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,2));
	deviceVariableCollection.print();

		deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();
}

__global__ void testParallel(){

	printf("/////////PARALLEL TEST/////////\n");

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,0));
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,0));
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,1));
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,2));
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,2));
	deviceVariableCollection.print();

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,3,
		deviceQueenPropagation.nextAssign(deviceVariableCollection,3));
	deviceVariableCollection.print();


}