#include <stdio.h>
#include "../../VariableCollection/VariableCollection.cu"
#include "../../QueenConstraints/QueenConstraints.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection deviceVariableCollection;
__device__ DeviceQueenConstraints deviceQueenConstraints;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int*,DeviceVariable*,int,int*,Triple*);
__global__ void print();

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void externParallelPropagation2(int var, int val){

	DeviceVariableCollection& vc = deviceVariableCollection; 

	int index = threadIdx.x +blockIdx.x*blockDim.x;
	int columnIndex = int((index % (vc.nQueen * vc.nQueen))%vc.nQueen);
	int rowIndex = int(((index % (vc.nQueen * vc.nQueen))/vc.nQueen) % vc.nQueen);

	if(index == 0)vc.deviceVariable[var].assign(val);
	
	if(index < vc.nQueen*vc.nQueen){
		if(rowIndex != var && val == columnIndex){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
		
		if(rowIndex != var && columnIndex == rowIndex && columnIndex+val-var < vc.nQueen && columnIndex+val-var >= 0){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+val-var],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
		
		if(rowIndex != var && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-val)+var < vc.nQueen && columnIndex-(vc.nQueen-val)+var >= 0){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-val)+var],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
	}
	__syncthreads();

	if(index == 0){
		int old = atomicAdd(&vc.deviceQueue.count,1);
		vc.deviceQueue.q[old].var = var;
		vc.deviceQueue.q[old].val = val;
		vc.deviceQueue.q[old].cs = 6;

	}

	if(index >= vc.nQueen && index < vc.nQueen*2)
		vc.deviceVariable[index-vc.nQueen].checkFailed();

	if(index >= vc.nQueen*2 && index < vc.nQueen*3)
		vc.deviceVariable[index-vc.nQueen*2].checkGround();

	bool ch = false;

	__syncthreads();


	do{
		
		__syncthreads();
		
		ch=false;
		
		for(int i = var+1; i < vc.nQueen; ++i){

			__syncthreads();

			if(vc.deviceVariable[i].changed == 1){

				__syncthreads();

				if(vc.deviceVariable[i].ground>=0){

					__syncthreads();

					if(index < vc.nQueen*vc.nQueen){
						if(rowIndex != i && vc.deviceVariable[i].ground == columnIndex){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
						
						if(rowIndex != i && columnIndex == rowIndex && columnIndex+vc.deviceVariable[i].ground-i < vc.nQueen && columnIndex+vc.deviceVariable[i].ground-i >= 0){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+vc.deviceVariable[i].ground-i],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
						
						if(rowIndex != i && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i < vc.nQueen && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i >= 0){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
					}

					__syncthreads();

					if(index == 0){
						int old = atomicAdd(&vc.deviceQueue.count,1);
						vc.deviceQueue.q[old].var = i;
						vc.deviceQueue.q[old].val = vc.deviceVariable[i].ground;
						vc.deviceQueue.q[old].cs = 6;

					}

					if(index >= vc.nQueen && index < vc.nQueen*2)
						vc.deviceVariable[index-vc.nQueen].checkFailed();

					if(index >= vc.nQueen*2 && index < vc.nQueen*3)
						vc.deviceVariable[index-vc.nQueen*2].checkGround();

					ch = true;
					__syncthreads();
				}

				vc.deviceVariable[i].changed=-1;
				__syncthreads();
			}

			__syncthreads();

		}
		__syncthreads();

	}while(ch);

}

////////////////////////////////////////////////////////////////////////////////////////////


int main(){
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);


	HostVariableCollection hostVariableCollection(30);
	init<<<1,1>>>(hostVariableCollection.dMem, 
				  hostVariableCollection.deviceVariableMem,
				  hostVariableCollection.nQueen,
				  hostVariableCollection.dMemlastValues,
				  hostVariableCollection.hostQueue.dMem);

	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(0,0);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(1,2);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(2,4);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(3,6);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(4,8);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(5,10);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(6,12);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(7,14);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(8,16);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(9,1);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(10,3);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(11,5);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(12,7);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(13,9);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(14,11);
	cudaDeviceSynchronize();
	externParallelPropagation2<<<1,30*30>>>(15,13);

	cudaDeviceSynchronize();
	print<<<1,1>>>();
	cudaDeviceSynchronize();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* dMem, DeviceVariable* deviceVariable, int nQueen, int* lv, Triple* q){
	deviceVariableCollection.init(deviceVariable,q,dMem,lv,nQueen);
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void print(){
	deviceVariableCollection.print();

}

////////////////////////////////////////////////////////////////////////////////////////////

