#pragma once
#include <stdio.h>

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostMemoryManagement{
	
	int* dMem;		//device memory ptr
	int rowSize;	//number of rows
	int colSize;	//number of columns
	int matSize;	//number of matrix 

	bool dbg;
	
	__host__ HostMemoryManagement(int,int,int);	//allocate memory
	__host__ int* getPtr();						//returns pointer to the memory
	__host__ ~HostMemoryManagement();			//free memory
};

///////////////////////////////////////////////////////////////////////

__host__ HostMemoryManagement::HostMemoryManagement(int nrw, int ncl, int nmt):
	rowSize(nrw),colSize(ncl),matSize(nmt),dbg(true){
		if(dbg)printf("\033[34mWarn\033[0m::HostMemoryManagement::constructor::ALLOCATION\n");
		cudaMalloc((void**)&dMem,sizeof(int)*rowSize*colSize*matSize);
}

///////////////////////////////////////////////////////////////////////

__host__ HostMemoryManagement::~HostMemoryManagement(){
	if(dbg)printf("\033[34mWarn\033[0m::HostMemoryManagement::destructor::DELLOCATION\n");
	cudaFree(dMem);
}

///////////////////////////////////////////////////////////////////////

__host__ int* HostMemoryManagement::getPtr()
	{return dMem;}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceMemoryManagement{

	int* dMem;		//device memory ptr
	int rowSize;	//number of rows
	int colSize;	//number of columns
	int matSize;	//number of matrix 

	bool dbg;

	__device__ DeviceMemoryManagement();					//do nothing
	__device__ ~DeviceMemoryManagement();					//do nothing
	__device__ DeviceMemoryManagement(int*,int,int,int);	//initialize
	__device__ void init(int*,int,int,int);					//initialize

	__device__ int get(int,int,int);	//returns as matrix element otherwise -1

	//////////////SINGLE THREAD//////////////
	__device__ int setSingle(int,int,int,int);				//returns 0 if element setting goes ok, otherwise -1
	__device__ int setRow(int,int,int);						//returns 0 if row setting goes ok, otherwise -1
	__device__ int setMatrix(int,int);						//returns 0 if matrix setting goes ok, otherwise -1
	__device__ int setFromTo(int,int,int,int,int,int,int);	//returns 0 if from-to setting goes ok, otherwise -1
															//take 3 index for starting element and 3 for end element
	__device__ void print();								//prints allMemoryStatus

	//////////////MULTI THREAD//////////////
	__device__ int setFromToMulti(int,int,int,int,int,int,int);			//returns 0 if from-to setting goes ok, otherwise -1
																		//take 3 index for starting element and 3 for end element
	__device__ int setMatrixFromToMultiLess(int,int,int);				//returns 0 of from-to settings goes ok, otherwise -1
																		//take index for start matrix and index for end matrix
	__device__ int copyMatrixFromToMulti(int,int,int);					//take index of copy matrix, index of start matrix of copy
																		//and index of end matrix of copy, returns 0 if ok, otherwise -1
	__device__ int copyMatrixFromToMultiLess(int,int,int);				//take index of copy matrix, index of start matrix of copy
																		//and index of end matrix of copy, returns 0 if ok, otherwise -1
																		//uses less thread
};

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceMemoryManagement::DeviceMemoryManagement(){}

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceMemoryManagement::~DeviceMemoryManagement(){}


///////////////////////////////////////////////////////////////////////

__device__ inline DeviceMemoryManagement::DeviceMemoryManagement(int* ptr, int nmt, int nrw, int ncl):
	rowSize(nrw),colSize(ncl),matSize(nmt),dMem(ptr),dbg(true){}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceMemoryManagement::init(int* ptr, int nmt, int nrw, int ncl){
	dMem = ptr;
	rowSize = nrw;
	colSize = ncl;
	matSize = nmt;
	dbg = true;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::get(int t, int i, int j){
	if(i < 0 || j < 0 || i > rowSize-1 || j > colSize-1 || t < 0 || t > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::get::INDEX OUT OF BOUND\n");
		return -1;
	}
	return dMem[rowSize*colSize*t + i*colSize + j];
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceMemoryManagement::print(){
	for(int t = 0; t < matSize; ++t){
		for(int i = 0; i < rowSize; ++i){
			for (int j = 0; j < colSize; ++j){
				printf("%d ", get(t,i,j));
			}printf("\n");
		}printf("\n");
	}printf("\n");
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::setSingle(int t,int i,int j,int value){
	if(i < 0 || j < 0 || i > rowSize-1 || j > colSize-1 || t < 0 || t > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setSingle::INDEX OUT OF BOUND\n");
		return -1;
	}

	dMem[rowSize*colSize*t + i*colSize + j] = value;
	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::setRow(int t, int i, int value){
	if(i < 0 || i > rowSize-1 || t < 0 || t > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setRow::INDEX OUT OF BOUND\n");
		return -1;
	}

	for(int k = rowSize*colSize*t+i*colSize; k < rowSize*colSize*t+(i+1)*colSize; ++k)
		dMem[k]=value;
	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::setMatrix(int t, int value){
	if(t < 0 || t > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setMatrix::INDEX OUT OF BOUND\n");
		return -1;
	}

	for(int k = rowSize*colSize*t; k < rowSize*colSize*(t+1); ++k)
		dMem[k] = value;

	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::setFromTo(int t0, int i0, int j0, int t1, int i1, int j1, int value){
	if(i0 < 0 || j0 < 0 || i0 > rowSize-1 || j0 > colSize-1 || t0 < 0 || t0 > matSize-1 || 
	   i1 < 0 || j1 < 0 || i1 > rowSize-1 || j1 > colSize-1 || t1 < 0 || t1 > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setFromTo::INDEX OUT OF BOUND\n");
		return -1;
	}

	for(int k = rowSize*colSize*t0+i0*colSize+j0; k < rowSize*colSize*t1+i1*colSize+j1; ++k)
		dMem[k]=value;

	return 0;
}

///////////////////////////////////////////////////////////////////////

__global__ void setExtern(int* dMem, int value, int n){
	if(threadIdx.x + blockIdx.x * blockDim.x < n)
		dMem[threadIdx.x + blockIdx.x * blockDim.x] = value;
}

__device__ inline int DeviceMemoryManagement::setFromToMulti(int t0, int i0, int j0, int t1, int i1, int j1, int value){
	if(i0 < 0 || j0 < 0 || i0 > rowSize-1 || j0 > colSize-1 || t0 < 0 || t0 > matSize-1 || 
	   i1 < 0 || j1 < 0 || i1 > rowSize-1 || j1 > colSize-1 || t1 < 0 || t1 > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setFromToMulti::INDEX OUT OF BOUND\n");
		return -1;
	}

	int numberOfElement = (rowSize*colSize*t1+i1*colSize+j1) - (rowSize*colSize*t0+i0*colSize+j0) + 1;
	if(numberOfElement <= 0){
		if(dbg)
			printf("\033[34mWarn\033[0m::DeviceMemoryManagement::setFromMulti::TO < FROM\n");
		return 0;
	}

	int numberOfThread = 1000;
	int numberOfBlock = int(numberOfElement/1000)+1;

	if(dbg){
		printf("\033[34mWarn\033[0m::DeviceMemoryManagement::");
		printf("setFromToMulti::DYNAMIC CALL OF \033[37m<<<%d,%d>>>\033[0m",numberOfBlock,numberOfThread);
		printf(" FOR %d ELEMENTS\n", numberOfElement);
	}

	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	setExtern<<<numberOfBlock,numberOfThread>>>(&dMem[rowSize*colSize*t0+i0*colSize+j0],value,numberOfElement);
	cudaStreamDestroy(s);

	cudaDeviceSynchronize();

	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::setMatrixFromToMultiLess(int from, int to, int value){
	if(from < 0 || to < 0 || from > matSize-1 || to > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::setMatrixFromToMultiLess::INDEX OUT OF BOUND\n");
		return -1;
	}

	int numberOfElement = (rowSize*colSize*(to+1)) - (rowSize*colSize*from);
	if(numberOfElement <= 0){
		if(dbg)
			printf("\033[34mWarn\033[0m::setMatrixFromToMultiLess::setFromMulti::TO < FROM\n");
		return 0;
	}

	int numberOfThread = 1000;
	int numberOfBlock = int(numberOfElement/1000)+1;

	if(dbg){
		printf("\033[34mWarn\033[0m::setMatrixFromToMultiLess::");
		printf("setMatrixFromToMultiLess::DYNAMIC CALL OF \033[37m<<<%d,%d>>>\033[0m",numberOfBlock,numberOfThread);
		printf(" FOR %d ELEMENTS\n", numberOfElement);
	}

	for(int i = 0; i < to+1-from; ++i){
		if(dbg){
			printf("\033[34mWarn\033[0m::setMatrixFromToMultiLess::");
			printf("setMatrixFromToMultiLess::DYNAMIC CALL OF \033[37m<<<%d,%d>>>\033[0m",numberOfBlock,numberOfThread);
			printf(" FOR %d ELEMENTS\n", numberOfElement);
		}

		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		setExtern<<<numberOfBlock,numberOfThread>>>(&dMem[rowSize*colSize*(from+i)],value,rowSize*colSize);
		cudaStreamDestroy(s);
	}

	cudaDeviceSynchronize();

	return 0;
}

///////////////////////////////////////////////////////////////////////

__global__ void copyExtern(int* from, int* to, int n, int size){
	if(threadIdx.x + blockIdx.x * blockDim.x < n)
		to[threadIdx.x + blockIdx.x * blockDim.x] = from[(threadIdx.x + blockIdx.x * blockDim.x)%size];
}

__device__ inline int DeviceMemoryManagement::copyMatrixFromToMulti(int what, int from, int to){
	if(what < 0 || from < 0 || to < 0 || what > matSize-1 || from > matSize-1 || to > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::copyMatrixFromToMulti::INDEX OUT OF BOUND\n");
		return -1;
	}

	int numberOfElement = (rowSize*colSize*(to+1)) - (rowSize*colSize*from);

	if(numberOfElement <= 0){
		if(dbg)
			printf("\033[34mWarn\033[0m::DeviceMemoryManagement::copyFromMulti::TO < FROM\n");
		return 0;
	}

	int numberOfThread = 1000;
	int numberOfBlock = int(numberOfElement/1000)+1;

	if(dbg){
		printf("\033[34mWarn\033[0m::DeviceMemoryManagement::");
		printf("copyMatrixFromToMulti::DYNAMIC CALL OF \033[37m<<<%d,%d>>>\033[0m",numberOfBlock,numberOfThread);
		printf(" FOR %d ELEMENTS\n", numberOfElement);
	}

	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	copyExtern<<<numberOfBlock,numberOfThread>>>(&dMem[rowSize*colSize*what],
												 &dMem[rowSize*colSize*from],
												 numberOfElement,rowSize*colSize);
	cudaStreamDestroy(s);

	cudaDeviceSynchronize();

	return 0;
}

///////////////////////////////////////////////////////////////////////

__device__ inline int DeviceMemoryManagement::copyMatrixFromToMultiLess(int what, int from, int to){
	if(what < 0 || from < 0 || to < 0 || what > matSize-1 || from > matSize-1 || to > matSize-1){
		printf("\033[31mError\033[0m::DeviceMemoryManagement::copyMatrixFromToMultiLess::INDEX OUT OF BOUND\n");
		return -1;
	}

	int numberOfThread = 0;
	int numberOfBlock = 0;
	int numberOfElement = (rowSize*colSize*(to+1)) - (rowSize*colSize*from);

	if(numberOfElement <= 0){
		if(dbg)
			printf("\033[34mWarn\033[0m::DeviceMemoryManagement::copyMatrixFromToMultiLess::TO < FROM\n");
		return 0;
	}

	numberOfBlock = int(rowSize*colSize/1000)+1;
	numberOfThread = 1000;

	for(int i = 0; i < to+1-from; ++i){
		if(dbg){
			printf("\033[34mWarn\033[0m::DeviceMemoryManagement::");
			printf("copyMatrixFromToMultiLess::DYNAMIC CALL OF \033[37m<<<%d,%d>>>\033[0m",numberOfBlock,numberOfThread);
			printf(" FOR %d ELEMENTS\n", rowSize * colSize);
		}

		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		copyExtern<<<numberOfBlock,numberOfThread>>>(&dMem[rowSize*colSize*what],
													 &dMem[rowSize*colSize*(from+i)],
													 rowSize * colSize, rowSize*colSize);
		cudaStreamDestroy(s);
	}

	cudaDeviceSynchronize();

	return 0;
}
