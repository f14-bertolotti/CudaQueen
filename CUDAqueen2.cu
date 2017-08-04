#include <stdio.h>
#include "./Variable/Variable.cu"
#include "./VariableCollection/VariableCollection.cu"
#include "./QueenConstraints/QueenConstraints.cu"
#include "./QueenPropagation/QueenPropagation.cu"
#include "./TripleQueue/TripleQueue.cu"
#include "./WorkSet/WorkSet.cu"
#include "./ErrorChecking/ErrorChecking.cu"
#include "./parallelQueue/parallelQueue.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__managed__ int nQueen = 8;
__managed__ int maxBlock = 1000;
__managed__ int levelDiscriminant1 = 4;
__managed__ int levelDiscriminant2 = 6;

__device__ int blockCount;
__device__ int nBlockInPhase2;
__device__ int printLock = 0;
__device__ int solutions;

__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceWorkSet deviceWorkSet;
__device__ DeviceParallelQueue deviceParallelQueue;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(int level, int workIndex){
		
	if(level == nQueen || deviceWorkSet.deviceVariableCollection[workIndex].isGround()){
		if(deviceQueenConstraints.solution(deviceWorkSet.deviceVariableCollection[workIndex],true)){
			atomicAdd(&solutions,1);
		}
	}else if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){
		//do nothing
	}else{

		if(level < levelDiscriminant1 && blockCount < maxBlock){

			int oldCount = 0;
			int nExpansions = deviceWorkSet.expand(workIndex,level,oldCount);
			if(nExpansions > 0){
				for(int i = oldCount; i < oldCount+nExpansions; ++i){
					cudaStream_t s;
					ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"main");
					test<<<1,1,0,s>>>(level+1,i);					
					atomicAdd(&blockCount,nExpansions);
					ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"main");
					ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"main");
				}
			}else{
				atomicAdd(&solutions,deviceWorkSet.solve(workIndex,level));
			}

		}else if(level < levelDiscriminant2){
			atomicAdd(&solutions,deviceWorkSet.solveAndAdd(workIndex,level,levelDiscriminant2,deviceParallelQueue));
		}else{
			atomicAdd(&solutions,deviceWorkSet.solve(workIndex,level));
		}
	}

	atomicAdd(&nBlockInPhase2,1);

	int levelLeaved = 0;
	do{
		levelLeaved = deviceParallelQueue.read(deviceWorkSet.deviceVariableCollection[workIndex],workIndex);
		if(levelLeaved == -1){
		}else{
			atomicAdd(&solutions,deviceWorkSet.solve(workIndex,levelLeaved));
		}
	}while(levelLeaved != -1);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initWorkSet( DeviceVariableCollection*,
							 DeviceVariable*,
							 int*,int*,Triple*,int,int);

__global__ void initParallelQueue(DeviceVariableCollection*,
								  DeviceVariable*,
								  int*,int*,int*,int*,Triple*,int,int);

__global__ void results();

////////////////////////////////////////////////////////////////////////////////////////////


int main(){

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeof(char)*999999999);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 20);

	HostWorkSet hostWorkSet(nQueen,maxBlock);

	HostParallelQueue hostParallelQueue(nQueen,maxBlock);

	initWorkSet<<<1,1>>>( hostWorkSet.deviceVariableCollection,
						  hostWorkSet.deviceVariable,
						  hostWorkSet.variablesMem,
						  hostWorkSet.lastValuesMem,
						  hostWorkSet.tripleQueueMem,
						  hostWorkSet.nQueen,
						  hostWorkSet.nVariableCollection);

	initParallelQueue<<<1,1>>>(hostParallelQueue.deviceVariableCollection,
				  			   hostParallelQueue.deviceVariable,
				 			   hostParallelQueue.variablesMem,
				 			   hostParallelQueue.lastValuesMem,
				 			   hostParallelQueue.lockReading,
				 			   hostParallelQueue.levelLeaved,
				 			   hostParallelQueue.tripleQueueMem,
				  			   hostParallelQueue.nQueen,
				 			   hostParallelQueue.size);



	cudaDeviceSynchronize();

    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
	float   elapsedTime;
	cudaEventRecord( start, 0 );

	test<<<1,1>>>(0,0);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	results<<<1,1>>>();

	printf("\033[36mTIME: %f\033[0m\n", elapsedTime);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void results(){


	printf("\033[32msolutions  = %d\033[0m\n",solutions);
	printf("block used = %d\n", deviceWorkSet.count);
	printf("block ended = %d\n", nBlockInPhase2);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initWorkSet( DeviceVariableCollection* deviceVariableCollection,
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

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initParallelQueue(DeviceVariableCollection* deviceVariableCollection, 
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

////////////////////////////////////////////////////////////////////////////////////////////


















