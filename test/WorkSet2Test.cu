#include "../WorkSet/WorkSet2.cu"
#include "../QueenConstraints/QueenConstraints2.cu"

__device__ DeviceWorkSet deviceWorkSet;
__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;

__global__ void init(DeviceVariableCollection*,
					 DeviceVariable*,
					 int*,int*,Triple*,int,int);
__global__ void test();

int main(){

	HostWorkSet hostWorkSet(8,11);
	init<<<1,1>>>(hostWorkSet.deviceVariableCollection,
				  hostWorkSet.deviceVariable,
				  hostWorkSet.variablesMem,
				  hostWorkSet.lastValuesMem,
				  hostWorkSet.tripleQueueMem,
				  hostWorkSet.nQueen,
				  hostWorkSet.nVariableCollection);
	cudaDeviceSynchronize();

	test<<<1,1>>>();

	return 0;
}


__global__ void init(DeviceVariableCollection* deviceVariableCollection,
					 DeviceVariable* deviceVariable,
					 int* variablesMem, int* lastValuesMem,
					 Triple* tripleQueueMem, int nQueen, int nVariableCollection){
	deviceWorkSet.init(deviceVariableCollection,
					   deviceVariable,
					   variablesMem,
					   lastValuesMem,
					   tripleQueueMem,
					   nQueen,
					   nVariableCollection);
}


__global__ void test(){
	int assignedValue = 0;
    assignedValue = deviceQueenPropagation.nextAssign(deviceWorkSet.deviceVariableCollection[0],0);
    deviceQueenPropagation.parallelForwardPropagation(deviceWorkSet.deviceVariableCollection[0],0,assignedValue);
    deviceWorkSet.expand(0,1);
    deviceWorkSet.expand(1,2);
    deviceWorkSet.expand(10,3);

	deviceWorkSet.print();
}