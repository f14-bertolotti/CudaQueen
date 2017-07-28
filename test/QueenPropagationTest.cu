#include <stdio.h>
#include "../QueenPropagation/QueenPropagation.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection1;
__device__ DeviceVariableCollection deviceVariableCollection2;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceQueenConstraints deviceQueenConstraints;

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init1(int*,DeviceVariable*,int,int*,Triple*);
__global__ void testSequential1();
__global__ void testParallel1();

///////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	HostVariableCollection hostVariableCollection1(6);
	init1<<<1,1>>>(hostVariableCollection1.dMem, 
				  hostVariableCollection1.deviceVariableMem,
				  hostVariableCollection1.nQueen,
				  hostVariableCollection1.dMemlastValues,
				  hostVariableCollection1.hostQueue.dMem);

	cudaDeviceSynchronize();
	testSequential1<<<1,1>>>();
	testParallel1<<<1,1>>>();

	cudaDeviceSynchronize();
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init1(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection1.init(deviceVariable,q,dMem,lv,nQueen);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void testSequential1(){

	printf("/////////SEQUENTIAL TEST/////////\n");

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,0));

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,1));

	deviceQueenPropagation.forwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

	printf("%s\n", deviceVariableCollection1.isFailed() ? "failed" : "not failed");

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void testParallel1(){

	printf("/////////PARALLEL TEST/////////\n");

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,0,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,0));

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,1,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,1));

	deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection1,2,
		deviceQueenPropagation.nextAssign(deviceVariableCollection1,2));
	deviceVariableCollection1.print();

	printf("%s\n", deviceVariableCollection1.isFailed() ? "failed" : "not failed");

	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
	deviceQueenPropagation.undoForwardPropagation(deviceVariableCollection1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Warn::HostQueue::constructor::ALLOCATION
Warn::HostVariableCollection::constructor::ALLOCATION
/////////SEQUENTIAL TEST/////////
[1] ::: 1 2 2 2 2 0  ::: grd:0 chd:-1 fld:-1 sz:6
[3] ::: 2 2 1 3 1 1  ::: grd:2 chd:-1 fld:-1 sz:6
[5] ::: 2 3 3 2 1 1  ::: grd:4 chd:-1 fld:-1 sz:6
[0] ::: 1 1 1 2 2 0  ::: grd:1 chd:-1 fld:-1 sz:6
[0] ::: 1 0 2 1 1 0  ::: grd:3 chd:-1 fld:-1 sz:6
[0] ::: 0 1 1 1 1 0  ::: grd:-1 chd:-1 fld:1 sz:6
(0,0,3)
(0,0,4)
(0,0,5)
(1,2,3)
(1,2,4)
(1,2,5)
(2,4,3)
(2,4,4)
(3,1,3)
(3,1,4)
(4,3,3)
(4,3,4)
(2,4,5)

failed
/////////PARALLEL TEST/////////
[2] ::: 2 1 2 1 1 0  ::: grd:1 chd:-1 fld:-1 sz:6
[4] ::: 3 3 2 1 2 0  ::: grd:3 chd:-1 fld:-1 sz:6
[1] ::: 1 2 3 3 2 0  ::: grd:0 chd:-1 fld:-1 sz:6
[0] ::: 0 2 1 1 1 1  ::: grd:2 chd:-1 fld:-1 sz:6
[0] ::: 1 1 1 1 1 0  ::: grd:4 chd:-1 fld:-1 sz:6
[0] ::: 1 0 0 2 1 0  ::: grd:-1 chd:-1 fld:1 sz:6
(0,1,6)
(0,1,5)
(1,3,6)
(1,3,5)
(2,0,6)
(3,2,6)
(4,4,6)
(2,0,5)

failed
Warn::HostVariableCollection::destructor::DELLOCATION
Warn::HostQueue::destructor::DELLOCATION
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////