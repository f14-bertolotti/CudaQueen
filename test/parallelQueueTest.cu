#include <stdio.h>
#include "../parallelQueue/parallelQueue.cu"
#include "../VariableCollection/VariableCollection.cu"

/////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceParallelQueue deviceParallelQueue;

__device__ const int nQueen = 8;
__device__ const int size = 1000;
__device__ int lockPrint = 0;

/////////////////////////////////////////////////////////////////

__global__ void init2(DeviceVariableCollection*,DeviceVariable*,
					 int*,int*,int*,int*,Triple*,int,int);
__global__ void init1(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();
__global__ void testPrint();

/////////////////////////////////////////////////////////////////

int main(){

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeof(char)*999999999);


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
	 			   hostParallelQueue.levelLeaved,
	 			   hostParallelQueue.tripleQueueMem,
	  			   hostParallelQueue.nQueen,
	 			   hostParallelQueue.size);

	cudaDeviceSynchronize();

	test<<<1,1>>>();

	cudaDeviceSynchronize();


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
		 			  int* levelLeaved,
		 			  Triple* tripleQueueMem,
		 			  int nQueen, 
		 			  int size){

	deviceParallelQueue.init(deviceVariableCollection,
						     deviceVariable,
						     variablesMem,
						     lastValuesMem,
						     lockReading,
						     levelLeaved,
						     tripleQueueMem,
						     nQueen,
						     size);

}

/////////////////////////////////////////////////////////////////

__global__ void test2(){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int levelLeaved = deviceParallelQueue.read(deviceVariableCollection,index);
	deviceParallelQueue.expansion(deviceVariableCollection,levelLeaved);

}

__global__ void test(){

	int index = threadIdx.x + blockIdx.x * blockDim.x;


	//deviceParallelQueue.add(deviceVariableCollection,0,index);

	//deviceParallelQueue.read(deviceVariableCollection,index);

	if(index == 0){
		deviceParallelQueue.expansion(deviceVariableCollection,0);
	}

	if(index == 0){
		deviceParallelQueue.expansion(deviceVariableCollection,1);
	}

	test2<<<1,1>>>();
	cudaDeviceSynchronize();
	test2<<<1,1>>>();
	cudaDeviceSynchronize();
	test2<<<1,1>>>();
	cudaDeviceSynchronize();

}

/////////////////////////////////////////////////////////////////

__global__ void testPrint(){
	deviceVariableCollection.print();
	deviceParallelQueue.printLocks();
	deviceParallelQueue.print();

}

/////////////////////////////////////////////////////////////////
