#include <stdio.h>
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceQueenConstraints deviceQueenConstraints;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();

////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	HostVariableCollection hostVariableCollection(8);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);
	cudaDeviceSynchronize();
	test<<<1,1>>>();
	cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
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
/*
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
Warn::HostVariableCollection::destructor::DELLOCATION
Warn::HostQueue::destructor::DELLOCATION
*/
////////////////////////////////////////////////////////////////////////////////////////////
