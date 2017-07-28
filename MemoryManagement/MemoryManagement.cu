#pragma once
#include "../ErrorChecking/ErrorChecking.cu"

////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct MemoryManagement{

	__device__ static inline void copy(T*,T*,unsigned int);	//copy from ptr to ptr for 

};

////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void externCopy(T* from, T* to, unsigned int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n) to[index] = from[index];
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ inline void MemoryManagement<T>::copy(T* from, T* to, unsigned int n){

	cudaStream_t stream;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"MemoryManagement<T>::copy::STREAM CREATION");
	externCopy<<<int(n/1000)+1,1000>>>(from,to,n);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"MemoryManagement<T>::copy::EXTERN COPY CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(stream),"MemoryManagement<T>::copy::STREAM DESTRUCTION");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"MemoryManagement<T>::copy::SYNCH");
}

////////////////////////////////////////////////////////////////////////////////////////////
