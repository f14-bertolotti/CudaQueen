#include <stdio.h>
#include "../QueenPropagation/QueenPropagation.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenConstraints/QueenConstraints.cu"
#include "../ErrorChecking/ErrorChecking.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init1(int*,DeviceVariable*,int,int*,Triple*);
__global__ void test();
__device__ int parallelForwardPropagation(DeviceVariableCollection&,int,int);
__global__ void parallelPropagation(DeviceVariableCollection&,int,int);

__device__ DeviceVariableCollection deviceVariableCollection; 
__managed__ int nQueen = 11;

///////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	HostVariableCollection hostVariableCollection1(nQueen);
	init1<<<1,1>>>(hostVariableCollection1.dMem, 
				  hostVariableCollection1.deviceVariableMem,
				  hostVariableCollection1.nQueen,
				  hostVariableCollection1.dMemlastValues,
				  hostVariableCollection1.hostQueue.dMem);

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init1(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	deviceVariableCollection.deviceVariable[0].assign(0);
	parallelForwardPropagation(deviceVariableCollection,0,0);
	cudaDeviceSynchronize();

	deviceVariableCollection.print();

	deviceVariableCollection.deviceVariable[1].assign(2);
	parallelForwardPropagation(deviceVariableCollection,1,2);
	cudaDeviceSynchronize();

	deviceVariableCollection.print();

	deviceVariableCollection.deviceVariable[2].assign(4);
	parallelForwardPropagation(deviceVariableCollection,2,4);
	cudaDeviceSynchronize();

	deviceVariableCollection.print();

	deviceVariableCollection.deviceVariable[3].assign(1);
	parallelForwardPropagation(deviceVariableCollection,3,1);
	cudaDeviceSynchronize();

	deviceVariableCollection.print();

	deviceVariableCollection.deviceVariable[4].assign(3);
	parallelForwardPropagation(deviceVariableCollection,4,3);
	cudaDeviceSynchronize();

	deviceVariableCollection.print();

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void parallelPropagation(DeviceVariableCollection& vc, int var, int val){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	int columnIndex = int((index % (vc.nQueen * vc.nQueen))%vc.nQueen);
	int rowIndex = int(((index % (vc.nQueen * vc.nQueen))/vc.nQueen) % vc.nQueen);

	if(rowIndex != var && val == columnIndex){

		int old = atomicSub(&vc.deviceVariable[rowIndex].domain[columnIndex],1);
		if(old == 1){
			vc.deviceVariable[rowIndex].changed = 1;
		}

	}
	
	if(rowIndex != var && columnIndex == rowIndex && columnIndex+val-var < vc.nQueen && columnIndex+val-var >= 0){

		int old = atomicSub(&vc.deviceVariable[rowIndex].domain[columnIndex+val-var],1);
		if(old == 1){
			vc.deviceVariable[rowIndex].changed = 1;
		}

	}
	
	if(rowIndex != var && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-val)+var < vc.nQueen && columnIndex-(vc.nQueen-val)+var >= 0){

		int old = atomicSub(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-val)+var],1);
		if(old == 1){
			vc.deviceVariable[rowIndex].changed = 1;
		}

	}

	__syncthreads();

	if(index == 0)
		vc.deviceQueue.add(var,val,6);

	if(index >= vc.nQueen && index < vc.nQueen*2)
		vc.deviceVariable[index-vc.nQueen].checkFailed();

	if(index >= vc.nQueen*2 && index < vc.nQueen*3)
		vc.deviceVariable[index-vc.nQueen*2].checkGround();
}

////////////////////////////////////////////////////////////////////////////

__device__ int parallelForwardPropagation(DeviceVariableCollection& vc, int var, int val){

	if(var < 0 || var > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAR OUT OF BOUND");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAL OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VARIABLE NOT GROUND");
		return -1;
	}

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelForwardPropagation::STREAM CREATION");

	parallelPropagation<<<1,vc.nQueen*vc.nQueen>>>(vc,var,val);

	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelForwardPropagation::EXTERN FORWARD PROPAGATION CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelForwardPropagation::STREAM DESTRUCTION");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelForwardPropagation::SYNCH");


	bool ch = false;

	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.deviceVariable[i].changed==1){
				if(vc.deviceVariable[i].ground>=0){

					cudaStream_t s;
					ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelForwardPropagation::STREAM CREATION");

					parallelPropagation<<<1,vc.nQueen*vc.nQueen>>>(vc,i,vc.deviceVariable[i].ground);

					ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelForwardPropagation::EXTERN FORWARD PROPAGATION CALL");
					ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelForwardPropagation::STREAM DESTRUCTION");


					ch = true;
				}
				vc.deviceVariable[i].changed=-1;
			}
		}
		ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelForwardPropagation::SYNCH");
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
