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

__managed__ int nQueen = 13;
__managed__ int maxBlock = 63000;
__managed__ int levelDiscriminant1 = 6;
__managed__ int levelDiscriminant2 = 10;

__device__ int blockCount = 0;
__device__ int nBlockInPhase2 = 0;
__device__ int printLock = 0;
__device__ int solutions = 0;

__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceWorkSet deviceWorkSet;
__device__ DeviceParallelQueue deviceParallelQueue;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(int level, int workIndex){


	if(level >= levelDiscriminant1 && !deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){
		atomicAdd(&solutions, deviceWorkSet.solveAndAdd(workIndex,level,levelDiscriminant2,deviceParallelQueue));
	}

	while(level < levelDiscriminant1){

		if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){

			break;

		}

		if(deviceWorkSet.deviceVariableCollection[workIndex].isGround() && 
			deviceQueenConstraints.solution(deviceWorkSet.deviceVariableCollection[workIndex],true)){
			
			atomicAdd(&solutions,1);
			break;

		}

		if(deviceWorkSet.deviceVariableCollection[workIndex].deviceVariable[level].ground < 0){
			int oldCount = 0;
			int nExpansion = deviceWorkSet.expand(workIndex,level,oldCount);
			if(nExpansion == -1){
				atomicAdd(&solutions, deviceWorkSet.solveAndAdd(workIndex,level,levelDiscriminant2,deviceParallelQueue));
				break;
			}else{
				for(int i = oldCount; i < oldCount + nExpansion; ++i){
					cudaStream_t s;
					ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"test::STREAM CREATION");
					test<<<1,1,0,s>>>(level+1,i);					
					atomicAdd(&blockCount,nExpansion);
					ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"test::TEST ERROR");
					ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"test::STREAM DESTRUCTION");
					
				}
			}
		}

		++level;

		if(level >= levelDiscriminant1 && !deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){
			atomicAdd(&solutions, deviceWorkSet.solveAndAdd(workIndex,level,levelDiscriminant2,deviceParallelQueue));
		}
	}

	atomicAdd(&nBlockInPhase2,1);

	int levelLeaved = 0;
	do{
		levelLeaved = deviceParallelQueue.read(deviceWorkSet.deviceVariableCollection[workIndex],workIndex);
		if(levelLeaved == -1){
		}else if(levelLeaved < levelDiscriminant2){
			atomicAdd(&solutions, deviceWorkSet.solveAndAdd(workIndex,levelLeaved,levelDiscriminant2,deviceParallelQueue));
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
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,65000);

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
	printf("start\n");

    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
	float   elapsedTime;
	cudaEventRecord( start, 0 );

	cudaStream_t s;
	ErrorChecking::hostErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"test::STREAM CREATION");
	test<<<1,1,0,s>>>(0,0);					
	ErrorChecking::hostErrorCheck(cudaStreamDestroy(s),"test::STREAM DESTRUCTION");
	ErrorChecking::hostErrorCheck(cudaDeviceSynchronize(),"test::SYNCH");
	ErrorChecking::hostErrorCheck(cudaPeekAtLastError(),"test::TEST ERROR");

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
	printf("still in queue = %d\n", deviceParallelQueue.stillInQueue());
	printf("maxUsed = %d\n", deviceParallelQueue.maxUsed);
	printf("block used = %d\n", deviceWorkSet.count);
	printf("block ended = %d\n", nBlockInPhase2);
	printf("block used real = %d\n", temp);
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


















