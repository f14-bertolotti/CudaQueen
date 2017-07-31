#include <stdio.h>
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceVariableCollection deviceVariableCollection2;
__device__ DeviceQueenConstraints deviceQueenConstraints;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void init2(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();
__global__ void testCopy();

////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);


	HostVariableCollection hostVariableCollection(8);
	HostVariableCollection hostVariableCollection2(8);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);
	init2<<<1,1>>>(hostVariableCollection2.dMem, 
				  hostVariableCollection2.deviceVariableMem,
				  hostVariableCollection2.nQueen,
				  hostVariableCollection2.dMemlastValues,
				  hostVariableCollection2.hostQueue.dMem);
	cudaDeviceSynchronize();
	test<<<1,1>>>();
	testCopy<<<1,1>>>();
	cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init2(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection2.init(deviceVariable,q,dMem,lv,nQueen);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){
	deviceVariableCollection.deviceQueue.add(1,1,1);
	deviceVariableCollection.deviceVariable[3].assign(3);
	deviceVariableCollection.deviceVariable[4].assign(3);
	deviceVariableCollection.deviceVariable[4].addTo(3,-1);
	deviceVariableCollection.print();
	printf("%s\n", deviceVariableCollection.isFailed() ? "is failed" : "not failed");
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void testCopy(){
	printf("///////////TEST OPERATOR =\\\\\\\\\\\\\\\n");
	deviceVariableCollection2 = deviceVariableCollection;
	deviceVariableCollection2.print();
}


////////////////////////////////////////////////////////////////////////////////////////////
/*
Warn::HostQueue::constructor::ALLOCATION
Warn::HostVariableCollection::constructor::ALLOCATION
Warn::HostQueue::constructor::ALLOCATION
Warn::HostVariableCollection::constructor::ALLOCATION
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 0 0 0 1 0 0 0 0  ::: grd:3 chd:-1 fld:-1 sz:8
[0] ::: 0 0 0 0 0 0 0 0  ::: grd:-1 chd:1 fld:1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
(1,1,1)

is failed
///////////TEST OPERATOR =\\\\\\\
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 0 0 0 1 0 0 0 0  ::: grd:3 chd:-1 fld:-1 sz:8
[0] ::: 0 0 0 0 0 0 0 0  ::: grd:-1 chd:1 fld:1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8
[0] ::: 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:8

Warn::HostVariableCollection::destructor::DELLOCATION
Warn::HostQueue::destructor::DELLOCATION
Warn::HostVariableCollection::destructor::DELLOCATION
Warn::HostQueue::destructor::DELLOCATION

*/
////////////////////////////////////////////////////////////////////////////////////////////
