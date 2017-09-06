#include <stdio.h>
#include <unistd.h>
#include "./Variable/Variable.cu"
#include "./VariableCollection/VariableCollection.cu"
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
__device__ int levelDiscriminant2 = 7;
__device__ bool fileprint = false;

__device__ int nodesPerBlock[100000];
__device__ int nodesPerBlockCount = 0;
__device__ bool activeNodesPerBlockCount = false;

int host_nQueen = 8;
int host_maxBlock = 10000;
int host_maxQueue = 10000;
int host_levelDiscriminant1 = 4;
int host_levelDiscriminant2 = 10;
bool host_fileprint = false;
bool host_activeNodesPerBlockCount = false;


__device__ int blockCount = 0;
__device__ int nBlockInPhase2 = 0;
__device__ int printLock = 0;
__device__ int solutions = 0;

__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceWorkSet deviceWorkSet;
__device__ DeviceParallelQueue deviceParallelQueue;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(int level, int workIndex,int countPerBlock){

	__shared__ bool done;

	done = false;

	__syncthreads();

	if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed())done = true;

	if(deviceWorkSet.deviceVariableCollection[workIndex].isGround()){
		if(threadIdx.x == 0){
			atomicAdd(&solutions,1);
		}
		done = true;
	}

	__syncthreads();

	while(!done){
		if(level < levelDiscriminant1){

			int nExpansion = 0;
			int oldCount = 0;

			nExpansion = deviceWorkSet.expand(workIndex,level,oldCount);
			if(threadIdx.x == 0)atomicAdd(&nodes,nExpansion+1);
			if(threadIdx.x == 0)nodesPerBlock[countPerBlock]+=nExpansion+1;
			__syncthreads();
			if(nExpansion > 0){
				if(threadIdx.x == 0){
				 	deviceWorkSet.deviceVariableCollection[workIndex].lastValues[level] = nQueen+1;
					for(int i = oldCount; i < oldCount + nExpansion; ++i){

					  	cudaStream_t s;
					 	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
					 	deviceWorkSet.deviceVariableCollection[i].lastValues[level] = nQueen+1;
						test<<<1,1024,	 sizeof(DeviceVariable)*nQueen
										+sizeof(Triple)*nQueen*nQueen*3
										+sizeof(int)*nQueen*nQueen
										+sizeof(int)*nQueen
										+sizeof(double)*10,
										s>>>(level+1,i,atomicAdd(&nodesPerBlockCount,1));

						cudaStreamDestroy(s);

					}
				}

				__syncthreads();

				if(deviceWorkSet.deviceVariableCollection[workIndex].isFailed())done = true;
				else if(deviceWorkSet.deviceVariableCollection[workIndex].isGround()){
					if(threadIdx.x == 0){
						atomicAdd(&solutions,1);
					}
					__syncthreads();
					done = true;
				}

				++level;

			}else if(nExpansion == -1){

				__syncthreads();

				if(maxQueue > 0){
					int tSol = deviceWorkSet.solveAndAdd(workIndex,level,nodes,countPerBlock,nodesPerBlock,levelDiscriminant2,deviceParallelQueue);
					if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
				}else{
					int tSol = deviceWorkSet.solve(workIndex,level,nodes,countPerBlock,nodesPerBlock);
					if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
				}

				done = true;
			}
				
		}else if(level >= levelDiscriminant1 && level < levelDiscriminant2){

			if(maxQueue > 0){
				int tSol = deviceWorkSet.solveAndAdd(workIndex,level,nodes,countPerBlock,nodesPerBlock,levelDiscriminant2,deviceParallelQueue);
				if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
			}else{
				int tSol = deviceWorkSet.solve(workIndex,level,nodes,countPerBlock,nodesPerBlock);
				if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
			}				

			done = true;

		}else{
			int tSol = deviceWorkSet.solve(workIndex,level,nodes,countPerBlock,nodesPerBlock);
			if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}
			done = true;
		}

		__syncthreads();

	}
	
	if(maxQueue > 0){
		int ll = 0;
		do{
			ll = deviceParallelQueue.read(deviceWorkSet.deviceVariableCollection[workIndex],workIndex);
			if(ll!=-1 && ll >= levelDiscriminant2){
				int tSol = deviceWorkSet.solve(workIndex,ll,nodes,countPerBlock,nodesPerBlock);
				if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}
			}else if(ll!=-1 && ll < levelDiscriminant2){
				if(maxQueue > 0){
					int tSol = deviceWorkSet.solveAndAdd(workIndex,ll,nodes,countPerBlock,nodesPerBlock,levelDiscriminant2,deviceParallelQueue);
					if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
				}else{
					int tSol = deviceWorkSet.solve(workIndex,ll,nodes,countPerBlock,nodesPerBlock);
					if(threadIdx.x == 0){atomicAdd(&solutions,tSol);}	
				}				
			}
			__syncthreads();
		}while(ll != -1);
	}

	return;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initWorkSet( DeviceVariableCollection*,
							 DeviceVariable*,
							 int*,int*,Triple*,int,int);

__global__ void initParallelQueue(DeviceVariableCollection*,
								  DeviceVariable*,
								  int*,int*,int*,int*,Triple*,int,int);

__global__ void results();

__global__ void initDevice(int,int,int,int,int,bool,bool);

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

	test<<<1,1024,	 sizeof(DeviceVariable)*host_nQueen
					+sizeof(Triple)*host_nQueen*host_nQueen*3
					+sizeof(int)*host_nQueen*host_nQueen
					+sizeof(int)*host_nQueen
					+sizeof(double)*10
					>>>(0,0,0);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaDeviceSynchronize();


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
	if(activeNodesPerBlockCount){
		printf("\n");
		for(int i = 0; i <= nodesPerBlockCount; ++i){
			printf("%d %d\n", i,nodesPerBlock[i]);
		}
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
	while ((c = getopt (argc, argv, "ofq:k:l:b:n:")) != -1){
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
			case 'o':
				host_activeNodesPerBlockCount = true;
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
	}

	initDevice<<<1,1>>>(host_nQueen,host_maxBlock,host_maxQueue,host_levelDiscriminant1,host_levelDiscriminant2,host_fileprint,host_activeNodesPerBlockCount);

	cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initDevice(int n,int b,int q,int l,int k,bool f,bool o){
	nQueen = n;
	maxBlock = b;
	maxQueue = q;
	levelDiscriminant1 = l;
	levelDiscriminant2 = k;

	fileprint = f;
	activeNodesPerBlockCount = o;
}

////////////////////////////////////////////////////////////////////////////////////////////












