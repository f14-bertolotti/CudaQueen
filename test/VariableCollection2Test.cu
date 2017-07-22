#include <stdio.h>
#include "../VariableCollection/VariableCollection2.cu"

__device__ DeviceVariableCollection deviceVariableCollection;

__global__ void init(int*,DeviceVariable*,int);
__global__ void test();


int main(){

	HostVariableCollection hostVariableCollection(8);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.dMemVariables,
				  hostVariableCollection.nQueen);

	test<<<1,1>>>();
}

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen){
	deviceVariableCollection.init(deviceVariable,dMem,nQueen);
}

__global__ void test(){
	deviceVariableCollection.print();
}