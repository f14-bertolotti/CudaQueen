#include <stdio.h>
#include "../../VariableCollection/VariableCollection.cu"
#include "../../QueenConstraints/QueenConstraints.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection to;
__device__ DeviceVariableCollection other;
__device__ DeviceQueenConstraints deviceQueenConstraints;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void init2(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();
__global__ void testCopy();

////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);


	HostVariableCollection hostVariableCollection(15);
	HostVariableCollection hostVariableCollection2(15);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);
	init2<<<1,1>>>(hostVariableCollection2.dMem, 
				  hostVariableCollection2.deviceVariableMem,
				  hostVariableCollection2.nQueen,
				  hostVariableCollection2.dMemlastValues,
				  hostVariableCollection2.hostQueue.dMem);
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();
	testCopy<<<1,1024>>>();
	cudaDeviceSynchronize();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	to.init(deviceVariable,q,dMem,lv,nQueen);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init2(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	other.init(deviceVariable,q,dMem,lv,nQueen);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void testCopy(){


	__shared__ int nQueen; 
	__shared__ int next1; 
	__shared__ int next2; 
	__shared__ int next3;

	nQueen = to.nQueen;
	
	next1 = ((((int(3*nQueen*nQueen/32)+1)*32)-3*nQueen*nQueen)+3*nQueen*nQueen);
	next2 = ((((int((next1+nQueen*nQueen)/32)+1)*32)-(next1+nQueen*nQueen))+(next1+nQueen*nQueen));
	next3 = ((((int((next2+nQueen)/32)+1)*32)-(next2+nQueen))+(next2+nQueen));

	if(threadIdx.x < 3*nQueen*nQueen){
		to.deviceQueue.q[threadIdx.x] = other.deviceQueue.q[threadIdx.x];
		to.deviceQueue.count = other.deviceQueue.count;
	}

	if(threadIdx.x >=  next1 && threadIdx.x < next1 + nQueen*nQueen){
		to.dMem[threadIdx.x - next1] = other.dMem[threadIdx.x - next1];
	}

	if(threadIdx.x >= next2 && threadIdx.x < next2 + nQueen){
		to.lastValues[threadIdx.x - next2] = other.lastValues[threadIdx.x- next2];
	}

	if(threadIdx.x >= next3 && threadIdx.x < next3 + nQueen){
		to.deviceVariable[threadIdx.x - next3].ground = other.deviceVariable[threadIdx.x - next3].ground;
		to.deviceVariable[threadIdx.x - next3].failed = other.deviceVariable[threadIdx.x - next3].failed;
		to.deviceVariable[threadIdx.x - next3].changed = other.deviceVariable[threadIdx.x - next3].changed;
	}

}


////////////////////////////////////////////////////////////////////////////////////////////
