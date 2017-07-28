#include <stdio.h>
#include "../TripleQueue/TripleQueue.cu"

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceQueue deviceQueue;

//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(Triple*,int);
__global__ void test();

//////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	HostQueue hostQueue(8);
	init<<<1,1>>>(hostQueue.dMem, hostQueue.nQueen);
	test<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(Triple* dMem, int nQueen){
	deviceQueue.init(dMem, nQueen);
}

//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	deviceQueue.add(1,1,1);
	deviceQueue.add(2,1,1);
	deviceQueue.add(3,1,1);
	deviceQueue.add(1,1,5);
	deviceQueue.add(1,1,1);
	deviceQueue.add(2,1,1);
	deviceQueue.add(3,1,1);				
	deviceQueue.add(1,1,5);
	Triple t = *deviceQueue.front();
	deviceQueue.print();
	printf("----%d,%d,%d----\n", t.var,t.val,t.cs);
	deviceQueue.pop();
	deviceQueue.pop();
	deviceQueue.pop();
	deviceQueue.pop();
	deviceQueue.print();

}

//////////////////////////////////////////////////////////////////////////////////////////////
/*
Warn::HostQueue::constructor::ALLOCATION
(1,1,1)
(2,1,1)
(3,1,1)
(1,1,5)
(1,1,1)
(2,1,1)
(3,1,1)
(1,1,5)
----1,1,5----
(1,1,1)
(2,1,1)
(3,1,1)
(1,1,5)
Warn::HostQueue::destructor::DELLOCATION
*/
//////////////////////////////////////////////////////////////////////////////////////////////
