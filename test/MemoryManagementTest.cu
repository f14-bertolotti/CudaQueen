#include <stdio.h>
#include "../MemoryManagement/MemoryManagement.cu"
#include "../ErrorChecking/ErrorChecking.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ const int n = 2048;
__device__ int a[n];
__device__ int b[n];

__global__ void test();

////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	test<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	for(int i = 0; i < n; ++i)
		a[i] = i;

	MemoryManagement<int>::copy(a,b,n);
	cudaDeviceSynchronize();


	for (int i = 0; i < n; ++i)
		printf("%d - %d \n", a[i], b[i]);

}

////////////////////////////////////////////////////////////////////////////////////////////
/*
0 - 0 
1 - 1 
2 - 2 
3 - 3 
4 - 4 
5 - 5 
6 - 6 
7 - 7 
8 - 8 
9 - 9 
10 - 10 
11 - 11 
12 - 12 
...
2046 - 2046 
2047 - 2047 
*/
////////////////////////////////////////////////////////////////////////////////////////////