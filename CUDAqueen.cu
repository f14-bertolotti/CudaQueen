#include <stdio.h>
#include "./Variable/Variable.cu"
#include "./VariableCollection/VariableCollection.cu"
#include "./QueenConstraints/QueenConstraints.cu"
#include "./QueenPropagation/QueenPropagation.cu"
#include "./TripleQueue/TripleQueue.cu"

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

__device__ const int nQueen=5;

__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;
__device__ DeviceVariableCollection deviceVariableCollection;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	int level = 0;
	int val = 0;
	int nSols = 0;
	bool done = false;
	do{
		//deviceVariableCollection.print();
		if(level == nQueen){
			if(deviceQueenConstraints.solution(deviceVariableCollection,true)){
				++nSols;
			}
			deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
			--level;			
		}else{

			val = deviceQueenPropagation.nextAssign(deviceVariableCollection,level);
			if(val == -1){
				if(level == 0) done = true;
				else{
					deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
					--level;
				}
			}else{
				deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection,level,val);
				if(deviceVariableCollection.isFailed()){
					deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection);
					--level;
				}
				++level;
			}
		}
	}while(!done);

	printf("\033[32mSOLUTIONS FOUND = %d\033[0m\n",nSols);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariable*,Triple*, int*,int*,int);

////////////////////////////////////////////////////////////////////////////////////////////

int main(){

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeof(char)*999999999);
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, nQueen*2);

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

	test<<<1,1>>>();
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\033[36mTIME: %f\033[0m\n", elapsedTime);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariable* variables,Triple* queue, int* varMem, int* lastValsMem, int nQueen){
	deviceVariableCollection.init(variables,queue,varMem,lastValsMem,nQueen);
	deviceVariableCollection.dbg = false;
}






















