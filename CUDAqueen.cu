#include <stdio.h>
#include "./Variable/Variable.cu"
#include "./TripleQueue/TripleQueue.cu"
#include "./VariableCollection/VariableCollection.cu"
#include "./QueenPropagation/QueenPropagation.cu"
#include "./QueenConstraints/QueenConstraints.cu"
#include "./ErrorChecking/ErrorChecking.cu"
#include "./parallelQueue/parallelQueue.cu"

////////////////////////////////////////////////////////////////////////////////////////////

/*
	nuovo cuda queen, utilizza le nuove strutture che permettono
	una esecuzione parallela di alcuni dei task che sono più
	dispensiosi dal punto di vista del tempo (utilizza chiamate dinamiche),
	i task in questione sono propagazione (sia in avanti che indietro),
	anche controllo dei vincoli.
	sei si alza troppo il valore di nQueen l'algoritmo potrebbe non funzionare
	a causa di un utilizzo maggior rispetto al consentito.
	il flusso di esecuzione principale è unico
*/

__device__ const int nQueen(8);

__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceVariableCollection deviceVariableCollection;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	__shared__ int level;
	__shared__ int levelUp;
	__shared__ int val;
	__shared__ int nSols;
	__shared__ bool done;

	level = 0;
	levelUp = 1;
	val = 0;
	nSols = 0;
	done = false;

	do{
		__syncthreads();
		if(deviceVariableCollection.isGround()){
			if(threadIdx.x == 0)++nSols;
			
			__syncthreads();
			deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
			if(threadIdx.x == 0){
				--level;
			}	
		}else{
			if(deviceVariableCollection.deviceVariable[level].ground < 0){
				__syncthreads();
				val = deviceQueenPropagation.nextAssign(deviceVariableCollection,level);
				__syncthreads();
				if(val == -1){
					if(level == 0){
						done = true;
					}else{
						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
						if(threadIdx.x == 0){
							level -= levelUp;
							levelUp = 1;
						}
					}
				}else{
					if(deviceQueenPropagation.parallelForwardPropagation2(deviceVariableCollection,level,val)){
						__syncthreads();
						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
						if(threadIdx.x == 0){
							--level;
						}
					}
					if(threadIdx.x == 0)++level;
				}
			}else{
				if(threadIdx.x == 0){
					++level;
					++levelUp;
				}
			}
		}

		__syncthreads();
	}while(!done);

	if(threadIdx.x == 0)printf("\033[32mSOLUTIONS FOUND = %d\033[0m\n",nSols);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariable*,Triple*, int*,int*,int);

////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	printf("NQUEEN %d\n", nQueen);

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeof(char)*999999999);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2);

	HostVariableCollection hostVariableCollection(nQueen);

	init<<<1,1>>>(hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.hostQueue.dMem,
				  hostVariableCollection.dMem,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.nQueen);
	cudaDeviceSynchronize();


    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
	float   elapsedTime;
	cudaEventRecord( start, 0 );

	test<<<1,1024>>>();
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\033[36mTIME: %f\033[0m\n", elapsedTime);

	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariable* variables,Triple* queue, int* varMem, int* lastValsMem, int nQueen){
	deviceVariableCollection.init(variables,queue,varMem,lastValsMem,nQueen);

}






















