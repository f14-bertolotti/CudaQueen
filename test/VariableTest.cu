#include <stdio.h>
#include "../Variable/Variable.cu"

//////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceVariable deviceVariable;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int*,int);
__global__ void test();

//////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	HostVariable hostVariable(10);
	init<<<1,1>>>(hostVariable.getPtr(),hostVariable.domainSize);
	test<<<1,1>>>();	

	cudaDeviceSynchronize();

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* domain, int domainSize){
	deviceVariable.init(domain,domainSize);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	deviceVariable.fullParallel = false;

	for(int i = 0; i < 10; ++i){
		deviceVariable.assign(i);
		deviceVariable.print();
		deviceVariable.undoAssign(i);
	}deviceVariable.print();

	deviceVariable.assign(0);
	deviceVariable.addTo(0,-1);
	deviceVariable.checkFailed();
	deviceVariable.checkGround();
	deviceVariable.print(); 

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Warn::HostVariable::HostVariable::ALLOCATION
1 0 0 0 0 0 0 0 0 0  ::: grd:0 chd:-1 fld:-1 sz:10
0 1 0 0 0 0 0 0 0 0  ::: grd:1 chd:-1 fld:-1 sz:10
0 0 1 0 0 0 0 0 0 0  ::: grd:2 chd:-1 fld:-1 sz:10
0 0 0 1 0 0 0 0 0 0  ::: grd:3 chd:-1 fld:-1 sz:10
0 0 0 0 1 0 0 0 0 0  ::: grd:4 chd:-1 fld:-1 sz:10
0 0 0 0 0 1 0 0 0 0  ::: grd:5 chd:-1 fld:-1 sz:10
0 0 0 0 0 0 1 0 0 0  ::: grd:6 chd:-1 fld:-1 sz:10
0 0 0 0 0 0 0 1 0 0  ::: grd:7 chd:-1 fld:-1 sz:10
0 0 0 0 0 0 0 0 1 0  ::: grd:8 chd:-1 fld:-1 sz:10
0 0 0 0 0 0 0 0 0 1  ::: grd:9 chd:-1 fld:-1 sz:10
1 1 1 1 1 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:10
0 0 0 0 0 0 0 0 0 0  ::: grd:-1 chd:1 fld:1 sz:10
Warn::HostVariable::~HostVariable::DEALLOCATION
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////
