#include <stdio.h>
#include "../parallelQueue/parallelQueue.cu"
#include "../VariableCollection/VariableCollection.cu"

/////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceParallelQueue deviceParallelQueue;

__device__ const int nQueen = 5;
__device__ const int size = 30;
__device__ int lockPrint = 0;

/////////////////////////////////////////////////////////////////

__global__ void init2(DeviceVariableCollection*,DeviceVariable*,
					 int*,int*,int*,Triple*,int,int);
__global__ void init1(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();
__global__ void testPrint();

/////////////////////////////////////////////////////////////////

int main(){

	HostParallelQueue hostParallelQueue(nQueen,size);
	HostVariableCollection hostVariableCollection(nQueen);

	init1<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);
	init2<<<1,1>>>(hostParallelQueue.deviceVariableCollection,
				  hostParallelQueue.deviceVariable,
				  hostParallelQueue.variablesMem,
				  hostParallelQueue.lastValuesMem,
				  hostParallelQueue.lockReading,
				  hostParallelQueue.tripleQueueMem,
				  hostParallelQueue.nQueen,
				  hostParallelQueue.size);

	cudaDeviceSynchronize();

	test<<<40,1>>>();

	testPrint<<<1,1>>>();

	cudaDeviceSynchronize();

	return 0;
}

/////////////////////////////////////////////////////////////////

__global__ void init1(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

/////////////////////////////////////////////////////////////////

__global__ void init2(DeviceVariableCollection* deviceVariableCollection, 
					 DeviceVariable* deviceVariable,
					 int* variablesMem,
					 int* lastValuesMem,
					 int* lockReading,
					 Triple* tripleQueueMem,
					 int nQueen, 
					 int size){

	deviceParallelQueue.init(deviceVariableCollection,
						     deviceVariable,
						     variablesMem,
						     lastValuesMem,
						     lockReading,
						     tripleQueueMem,
						     nQueen,
						     size);

}

/////////////////////////////////////////////////////////////////

__global__ void test(){

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	while(atomicCAS(&lockPrint,0,1)==1){}
		printf("%d adding\n", index);
	lockPrint = 0;

	if(deviceParallelQueue.add(deviceVariableCollection)==-1){
		while(atomicCAS(&lockPrint,0,1)==1){}
			printf("%d cant add\n", index);
		lockPrint = 0;
	}

	while(atomicCAS(&lockPrint,0,1)==1){}
		printf("%d reading\n", index);
	lockPrint = 0;

	if(deviceParallelQueue.read(deviceVariableCollection)==-1){
		while(atomicCAS(&lockPrint,0,1)==1){}
			printf("%d cant read\n", index);
		lockPrint = 0;
	}

	cudaDeviceSynchronize();

}

/////////////////////////////////////////////////////////////////

__global__ void testPrint(){
	deviceParallelQueue.print();
}

/////////////////////////////////////////////////////////////////
