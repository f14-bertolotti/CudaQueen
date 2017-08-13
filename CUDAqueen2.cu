#include <stdio.h>
#include <unistd.h>
#include "./Variable/Variable.cu"
#include "./VariableCollection/VariableCollection.cu"
#include "./QueenConstraints/QueenConstraints.cu"
#include "./QueenPropagation/QueenPropagation.cu"
#include "./TripleQueue/TripleQueue.cu"
#include "./WorkSet/WorkSet.cu"
#include "./ErrorChecking/ErrorChecking.cu"
#include "./parallelQueue/parallelQueue.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ int nQueen = 8;
__device__ int maxBlock = 10000;
__device__ int maxQueue = 10000;
__device__ int nodes = 1;
__device__ int levelDiscriminant1 = 4;
__device__ int levelDiscriminant2 = 10;
__device__ bool fileprint = false;

int host_nQueen = 8;
int host_maxBlock = 10000;
int host_maxQueue = 10000;
int host_levelDiscriminant1 = 4;
int host_levelDiscriminant2 = 10;
bool host_fileprint = false;

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

	/*while(atomicCAS(&printLock,0,1)==1){}
	deviceWorkSet.deviceVariableCollection[workIndex].print();
	printLock = 0;*/


	bool done = false;

	if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){
		done = true;
	}else if(deviceWorkSet.deviceVariableCollection[workIndex].isGround()){
		if(deviceQueenConstraints.solution(deviceWorkSet.deviceVariableCollection[workIndex],true)){
			atomicAdd(&solutions,1);
			done = true;
		}
	}

	while(!done){

		if(level < levelDiscriminant1){

			//espansione
			int nExpansion = 0;
			int oldCount = 0;
			nExpansion = deviceWorkSet.expand(workIndex,level,oldCount);

			atomicAdd(&nodes,nExpansion+1);
			if(nExpansion >= 0){
				//sono riuscito ad espandere
				//deviceWorkSet.deviceVariableCollection[workIndex].deviceQueue.count = 0;

				for(int i = oldCount; i < oldCount + nExpansion; ++i){

				  	cudaStream_t s;
				 	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
				 	deviceWorkSet.deviceVariableCollection[i].lastValues[level] = nQueen+1;
				 	//deviceWorkSet.deviceVariableCollection[i].deviceQueue.count = 0;
					test<<<1,1,0,s>>>(level+1,i);
					cudaStreamDestroy(s);

				}

				if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed()){
					done = true;
				}else if(deviceWorkSet.deviceVariableCollection[workIndex].isGround()){
					if(deviceQueenConstraints.solution(deviceWorkSet.deviceVariableCollection[workIndex],true)){
						atomicAdd(&solutions,1);
					}
					done = true;
				}

				++level;
			}else{
				//non sono riuscito ad espandere risolvo normalmente
				if(maxQueue > 0){
					atomicAdd(&solutions,deviceWorkSet.solveAndAdd(workIndex,level,nodes,levelDiscriminant2,deviceParallelQueue));
				}else{
					atomicAdd(&solutions,deviceWorkSet.solve(workIndex,level,nodes));
				}
				done = true;
			}

		}else if(level >= levelDiscriminant1 && level < levelDiscriminant2){

			if(maxQueue > 0){
				atomicAdd(&solutions,deviceWorkSet.solveAndAdd(workIndex,level,nodes,levelDiscriminant2,deviceParallelQueue));
			}else{
				atomicAdd(&solutions,deviceWorkSet.solve(workIndex,level,nodes));
			}
			done = true;

		}else{

			atomicAdd(&solutions,deviceWorkSet.solve(workIndex,level,nodes));
			done = true;

		}

	}


	int levelLeaved = 0;
	do{
		levelLeaved = deviceParallelQueue.read(deviceWorkSet.deviceVariableCollection[workIndex],workIndex);
		if(levelLeaved == -1){
		}else if(levelLeaved < levelDiscriminant2){
			if(maxQueue > 0){
				atomicAdd(&solutions,deviceWorkSet.solveAndAdd(workIndex,levelLeaved,nodes,levelDiscriminant2,deviceParallelQueue));
			}else{
				atomicAdd(&solutions,deviceWorkSet.solve(workIndex,levelLeaved,nodes));
			}
		}else{
			atomicAdd(&solutions,deviceWorkSet.solve(workIndex,levelLeaved,nodes));
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

__global__ void init(int,int,int,int,int,bool);

void init(int argc, char **argv);


////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv){

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeof(char)*999999999);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 20);
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,65000);

	init(argc, argv);

	HostWorkSet hostWorkSet(host_nQueen,host_maxBlock);

	HostParallelQueue hostParallelQueue(host_nQueen,host_maxQueue);

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

	cudaDeviceSynchronize();

	if(!host_fileprint)printf("\033[36mTIME: %f\033[0m\n", elapsedTime);
	else printf("%f\n", elapsedTime);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void results(){

	//deviceWorkSet.print();
	if(!fileprint){
		printf("\033[32msolutions  = %d\033[0m\n",solutions);
		printf("still in queue = %d\n", deviceParallelQueue.stillInQueue());
		printf("maxUsed = %d\n", deviceParallelQueue.maxUsed);
		printf("block used = %d\n", deviceWorkSet.count);
	}else{
		printf("%d ",nQueen);
		printf("%d ",solutions);
		printf("%d ",nodes);
		printf("%d ",deviceWorkSet.count);
		printf("%d ",deviceParallelQueue.maxUsed);
		printf("%d ",maxBlock);
		printf("%d ",maxQueue);
		printf("%d ",levelDiscriminant1);
		printf("%d ",levelDiscriminant2);
	}
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

void init(int argc, char **argv){
	char c;
	while ((c = getopt (argc, argv, "fq:k:l:b:n:")) != -1){
		switch (c){
			case 'n':
				host_nQueen = atoi(optarg);
				break;
			case 'b':
				host_maxBlock = atoi(optarg);
				break;
			case 'l':
				host_levelDiscriminant1 = atoi(optarg);
				break;
			case 'k':
				host_levelDiscriminant2 = atoi(optarg);
				break;
			case 'q':
				host_maxQueue = atoi(optarg);
				break;
			case 'f':
				host_fileprint = true;
				break;
			default:
				abort();
		}
	}

	if(host_fileprint == false){
		printf("-------------------------\n");
		printf("number of queen      = %d\n", host_nQueen);
		printf("max number of block  = %d\n", host_maxBlock);
		printf("level discriminant 1 = %d\n", host_levelDiscriminant1);
		printf("level discriminant 2 = %d\n", host_levelDiscriminant2);
		printf("file print           = %s\n", host_fileprint ? "true" : "false");
		printf("-------------------------\n");
	}else

	init<<<1,1>>>(host_nQueen,host_maxBlock,host_maxQueue,host_levelDiscriminant1,host_levelDiscriminant2,host_fileprint);
	cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int n,int b,int q,int l,int k,bool f){
	nQueen = n;
	maxBlock = b;
	maxQueue = q;
	levelDiscriminant1 = l;
	levelDiscriminant2 = k;
	fileprint = f;
}

////////////////////////////////////////////////////////////////////////////////////////////












