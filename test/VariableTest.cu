#include <stdio.h>
#include "../Variable/Variable.cu"

__device__ DeviceVariable dv;

__global__ void init(int*,int);
__global__ void test();


int main(){

	HostVariable hv(10);
	init<<<1,1>>>(hv.getPtr(),hv.DomainSize);
	test<<<1,1>>>();	


	cudaDeviceSynchronize();

	return 0;
}



__global__ void init(int* domain, int domainSize){
	dv.init(domain,domainSize);
}


__global__ void test(){

	dv.fullParallel = false;

	for(int i = 0; i < 10; ++i){
		dv.assign(i);
		dv.print();
		dv.undoAssign(i);
	}dv.print();

	dv.assign(0);
	dv.addTo(0,-1);
	dv.checkFailed();
	dv.checkGround();
	dv.print(); 

}