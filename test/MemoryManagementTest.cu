#include <stdio.h>
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////

__device__ DeviceMemoryManagement deviceMemoryManagement;

__global__ void init(int*,int,int,int);
__global__ void test();

///////////////////////////////////////////////////////////////////

int main(){

	HostMemoryManagement hostMemoryManagement(5,11,100);	
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
	deviceMemoryManagement.setMatrixFromToMultiLess(0,4,1);
	deviceMemoryManagement.setMatrixFromToMultiLess(3,4,0);

	deviceMemoryManagement.setFromToMulti(0,0,0,4,4,9,0);
	deviceMemoryManagement.setFromToMulti(0,0,0,0,4,0,2);
/*	deviceMemoryManagement.setMatrix(2,3);
	deviceMemoryManagement.setRow(2,2,4);
	deviceMemoryManagement.setSingle(2,3,0,5);
*/	deviceMemoryManagement.copyMatrixFromToMulti(2,0,3);
	deviceMemoryManagement.copyMatrixFromToMulti(0,0,4);
	deviceMemoryManagement.setFromToMulti(0,0,0,1,4,9,1);
	deviceMemoryManagement.copyMatrixFromToMulti(2,3,2);
	deviceMemoryManagement.copyMatrixFromToMultiLess(4,0,0);
	deviceMemoryManagement.copyMatrixFromToMultiLess(4,0,0);
	deviceMemoryManagement.copyMatrixFromToMultiLess(1,0,4);
	deviceMemoryManagement.setMatrixFromToMultiLess(0,4,4);

	deviceMemoryManagement.print();

}

///////////////////////////////////////////////////////////////////
