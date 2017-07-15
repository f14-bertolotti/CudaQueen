#include <stdio.h>

///////////////////////////////////////////////////////////////////////

__device__ const int nQueen	  = 5;
__device__ const int nThreads = 5;

///////////////////////////////////////////////////////////////////////

__device__ int get(int*,int,int);
__device__ int set(int*,int,int,int);
__device__ int decrement(int*,int,int);
__device__ int increment(int*,int,int);
__global__ void parallelAllDiffs(int*,int);
__global__ void parallelDiagConstr(int*,int);
__global__ void copy(int*,int*);
__global__ void printAll(int*);
__global__ void preparePropagation(int*,int);
__global__ void propagation(int*,int);
__global__ void init(int*);

///////////////////////////////////////////////////////////////////////

int main(){

	int* HostQueenMem;
	int* DeviceQueenMem;

	HostQueenMem =   (int*)malloc(sizeof(int)*nQueen*nQueen*nThreads);
	cudaMalloc((void**)&DeviceQueenMem,sizeof(int)*nQueen*nQueen*nThreads);

	init<<<1,1>>>(DeviceQueenMem);

	preparePropagation<<<1,nThreads>>>(DeviceQueenMem,0);
	propagation<<<1,nThreads>>>(DeviceQueenMem,0);
	printAll<<<1,1>>>(DeviceQueenMem);
	cudaDeviceSynchronize();

	printf("-------Round 2--------\n");

	copy<<<1,nQueen*nQueen*nThreads>>>(DeviceQueenMem,DeviceQueenMem);
	preparePropagation<<<1,nThreads>>>(DeviceQueenMem,1);
	propagation<<<1,nThreads>>>(DeviceQueenMem,1);
	printAll<<<1,1>>>(DeviceQueenMem);
	cudaDeviceSynchronize();

	printf("-------Round 3--------\n");

	copy<<<1,nQueen*nQueen*nThreads>>>(&DeviceQueenMem[nQueen*nQueen*2],DeviceQueenMem);
	preparePropagation<<<1,nThreads>>>(DeviceQueenMem,2);
	propagation<<<1,nThreads>>>(DeviceQueenMem,2);
	printAll<<<1,1>>>(DeviceQueenMem);
	cudaDeviceSynchronize();

	printf("-------Round 4--------\n");

	copy<<<1,nQueen*nQueen*nThreads>>>(&DeviceQueenMem[nQueen*nQueen*4],DeviceQueenMem);
	preparePropagation<<<1,nThreads>>>(DeviceQueenMem,3);
	propagation<<<1,nThreads>>>(DeviceQueenMem,3);
	printAll<<<1,1>>>(DeviceQueenMem);
	cudaDeviceSynchronize();

	printf("-------checkSolutions--------\n");
	printAll<<<1,1>>>(DeviceQueenMem);

	parallelAllDiffs<<<1,nThreads>>>(DeviceQueenMem,1);
	parallelDiagConstr<<<1,nThreads*4>>>(DeviceQueenMem,1);

	free(HostQueenMem);
	cudaFree(DeviceQueenMem);
	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ int get(int* Mem, int thread, int i, int j){
	if(i < 0 || j < 0 || i > nQueen-1 || j > nQueen-1){
		printf("getError out of bounds\n");
		return -1;
	}
	return Mem[nQueen*nQueen*thread + i*nQueen + j];
}

///////////////////////////////////////////////////////////////////////

__device__ int set(int* Mem, int thread, int i, int j, int value){
	if(i < 0 || j < 0 || i > nQueen-1 || j > nQueen-1){
		printf("setError out of bounds\n");
		return -1;
	}
	Mem[nQueen*nQueen*thread + i*nQueen + j] = value;
	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ int increment(int* Mem, int thread, int i, int j){
	return set(Mem,thread,i,j,get(Mem,thread,i,j)+1);
}

///////////////////////////////////////////////////////////////////////

__device__ int decrement(int* Mem, int thread, int i, int j){
	return set(Mem,thread,i,j,get(Mem,thread,i,j)-1);
}

///////////////////////////////////////////////////////////////////////

__device__ bool okAllDiffs = true;
__global__ void parallelAllDiffs(int* Mem,int nMatrix){
	int sum = 0;
	for(int i = 0 ; i < nQueen; ++i)
		if(get(Mem,nMatrix,i,threadIdx.x)==1)
			++sum;
	
	if(sum != 1)
		okAllDiffs = false;

	//printf("sum : %d\n", sum);

	__syncthreads();

	if(threadIdx.x == 0){
		if(okAllDiffs)printf("ok AllDiff\n");
		else printf("not ok AllDiff\n");
		okAllDiffs = true;
	}
}

///////////////////////////////////////////////////////////////////////
__device__ bool okDiags = true;
__global__ void parallelDiagConstr(int* Mem, int nMatrix){
	int sum,i,j,what;

	if(threadIdx.x < nQueen)what = 0;
	else if(threadIdx.x >= nQueen && threadIdx.x<2*nQueen)what = 1;
	else if(threadIdx.x >= 2*nQueen && threadIdx.x<3*nQueen)what = 2;
	else if(threadIdx.x >= 3*nQueen && threadIdx.x<4*nQueen)what = 3;

	switch(what){
		case 0:{
			j = threadIdx.x % nQueen;
			i = 0;
			sum = 0;
			while(j < nQueen && i < nQueen){
				if(get(Mem,nMatrix,i,j)==1)++sum;
				++j;
				++i;
			}
			if(sum > 1)
				okDiags = false;

			//printf("idx %d, sum0 : %d\n",threadIdx.x % nQueen, sum);
					
			break;
		}
		case 1:{

			i = threadIdx.x % nQueen;
			j = 0;
			sum = 0;
			while(j < nQueen && i < nQueen){
				if(get(Mem,nMatrix,i,j)==1)++sum;
				++j;
				++i;
			}
	
			if(sum > 1)
				okDiags = false;

			//printf("idx %d, sum1 : %d\n",threadIdx.x % nQueen, sum);

			break;
		}
		case 2:{

			j = threadIdx.x % nQueen;
			i = 0;
			sum = 0;
			while(j >= 0 && i < nQueen){
				if(get(Mem,nMatrix,i,j)==1)++sum;
				--j;
				++i;
			}

			if(sum > 1)
				okDiags = false;

			//printf("idx %d, sum2 : %d\n",threadIdx.x % nQueen, sum);

			break;
		}
		case 3:{
			i = threadIdx.x % nQueen;
			j = nQueen-1;
			sum = 0;
			while(j >= 0 && i < nQueen){
				if(get(Mem,nMatrix,i,j)==1)++sum;
				--j;
				++i;
			}
			if(sum > 1)
				okDiags = false;

			//printf("idx %d, sum3 : %d\n",threadIdx.x % nQueen, sum);

			break;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		if(okDiags)printf("ok Diags\n");
		else printf("not ok Diags\n");
		okDiags = true;
	}

}

///////////////////////////////////////////////////////////////////////

__global__ void copy(int* from, int* to){
	to[threadIdx.x] = from[threadIdx.x%(nQueen*nQueen)];
}

///////////////////////////////////////////////////////////////////////

__global__ void init(int* Mem){

	for(int t = 0; t < nThreads; ++t){
		for(int i = 0; i < nQueen; ++i){
			for (int j = 0; j < nQueen; ++j){
				set(Mem,t,i,j,1);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////

__global__ void printAll(int* Mem){
	for(int t = 0; t < nThreads; ++t){
		for(int i = 0; i < nQueen; ++i){
			for (int j = 0; j < nQueen; ++j){
				printf("%d ",get(Mem,t,i,j) <= 0 ? 0: 1);			
			}printf("\n");
		}printf("\n");
	}printf("\n");
}

///////////////////////////////////////////////////////////////////////

__global__ void preparePropagation(int* Mem, int i){
	for(int j = 0; j < nQueen; ++j)
		if(j != threadIdx.x)
			decrement(Mem,threadIdx.x,i,j);
}

///////////////////////////////////////////////////////////////////////

__global__ void propagation(int* Mem,int var){

	if(get(Mem,threadIdx.x,var,threadIdx.x)==1){

		for(int i = 0; i < nQueen; ++i)
			if(i != var){
				decrement(Mem,threadIdx.x,i,threadIdx.x);
			}

		int i=var+1,j=threadIdx.x+1;
		while(i<nQueen && j<nQueen){
			decrement(Mem,threadIdx.x,i,j);
			++i;++j;
		}

		i=var-1,j=threadIdx.x-1;
		while(i>=0 && j>=0){
			decrement(Mem,threadIdx.x,i,j);
			--i;--j;
		}

		i=var-1,j=threadIdx.x+1;
		while(i>=0 && j<nQueen){
			decrement(Mem,threadIdx.x,i,j);
			--i;++j;
		}

		i=var+1,j=threadIdx.x-1;
		while(i<nQueen && j>=0){
			decrement(Mem,threadIdx.x,i,j);
			++i;--j;
		}
	}
}

///////////////////////////////////////////////////////////////////////
