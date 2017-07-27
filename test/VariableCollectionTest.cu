#include <stdio.h>
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"


__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceQueenConstraints deviceQueenConstraints;


__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();


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

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

__global__ void test(){
	deviceVariableCollection.deviceQueue.add(1,1,1);
	deviceVariableCollection.print();
}