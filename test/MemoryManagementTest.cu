#include <stdio.h>
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////

__device__ DeviceMemoryManagement deviceMemoryManagement;

__global__ void init(int*,int,int,int);
__global__ void test();

///////////////////////////////////////////////////////////////////

int main(){

	HostMemoryManagement hostMemoryManagement(5,5,10);	
	init<<<1,1>>>(hostMemoryManagement.dMem, hostMemoryManagement.rowSize,
				  hostMemoryManagement.colSize, hostMemoryManagement.matSize);

	test<<<1,1>>>();

	cudaDeviceSynchronize();

	return 0;
}


///////////////////////////////////////////////////////////////////

__global__ void init(int* dMem, int rowSize, int colSize, int matSize){
	deviceMemoryManagement.init(dMem,rowSize,colSize,matSize);
}

///////////////////////////////////////////////////////////////////

__global__ void test(){
	deviceMemoryManagement.setFromToMulti(0,0,0,4,4,9,0);
	deviceMemoryManagement.setFromToMulti(0,0,0,0,4,0,2);
	deviceMemoryManagement.setMatrix(2,3);
	deviceMemoryManagement.setRow(2,2,4);
	deviceMemoryManagement.setSingle(2,3,0,5);
	deviceMemoryManagement.copyFromToMulti(2,0,3);
	deviceMemoryManagement.copyFromToMulti(0,0,4);
	deviceMemoryManagement.setFromToMulti(0,0,0,1,4,9,1);
	deviceMemoryManagement.copyFromToMulti(2,3,2);

	deviceMemoryManagement.print();

}

///////////////////////////////////////////////////////////////////
