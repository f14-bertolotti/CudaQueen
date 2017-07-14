#include <stdio.h>
#include "./parallelQueue.cu"

/////////////////////////////////////////////////////////////////

__device__ struct test{
	int x;
	__device__ void print(){printf("%d\n",x);}
	__device__ void copy(test& t){x = t.x;}
};

/////////////////////////////////////////////////////////////////

__device__ parallelQueue<test> q;
__device__ const int n = 10;

/////////////////////////////////////////////////////////////////

__global__ void init(test*,int*);
__global__ void testAdd();
__global__ void testFrontAndPop();
__global__ void testAddAndFrontAndPop();
__global__ void testPop();
__global__ void testPrint();

/////////////////////////////////////////////////////////////////

int main(){

	int* pQueueLockMem; 
	test* pQueueMem;

	cudaMalloc((void**)&pQueueLockMem,sizeof(int)*n);
	cudaMalloc((void**)&pQueueMem,sizeof(test)*n);

	init<<<1,1>>>(pQueueMem,pQueueLockMem);

	testAdd<<<n,1>>>();
	cudaDeviceSynchronize();

	testPop<<<n,1>>>();
	cudaDeviceSynchronize();

	testPrint<<<1,1>>>();
	cudaDeviceSynchronize();

	testAddAndFrontAndPop<<<n,1>>>();
	cudaDeviceSynchronize();


	cudaFree(pQueueLockMem);
	cudaFree(pQueueMem);

	return 0;
}


/////////////////////////////////////////////////////////////////

__global__ void init(test* pMem0, int* pMem1){
	q.init(pMem0,pMem1,0,n);

	for(int i = 0; i < n; ++i)
		pMem1[i]=0;

	printf("\n");
}

/////////////////////////////////////////////////////////////////

__global__ void testAdd(){
	if(blockIdx.x == 1)printf("TEST ADD\n");

	test a;
	a.x=blockIdx.x;

	q.add(a);
}

/////////////////////////////////////////////////////////////////

__global__ void testFrontAndPop(){
	if(blockIdx.x == 1)printf("TEST FRONT\n");
	test a;

	q.frontAndPop(a);
}

/////////////////////////////////////////////////////////////////

__global__ void testAddAndFrontAndPop(){
	if(blockIdx.x == 1)printf("TEST ADD AND FRONT\n");
	test a;
	a.x=blockIdx.x;

	q.add(a);
	printf("idx %d added\n",blockIdx.x);
	q.frontAndPop(a);
	printf("idx %d read ",blockIdx.x);a.print();
}

/////////////////////////////////////////////////////////////////

__global__ void testPop(){

	if(blockIdx.x == 1)printf("TEST POP\n");

	q.pop();
}

/////////////////////////////////////////////////////////////////

__global__ void testPrint(){
	printf("TEST PRINT\n");
	q.print();
}

/////////////////////////////////////////////////////////////////
