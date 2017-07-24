#pragma once
#include <stdio.h>
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariable{
	
	int* dMem;						//ptr to memory
	int DomainSize;					//variable size (cardinality)
	bool dbg;						//verbose

	__host__ HostVariable(int); 	//allocate memory using HostMemoryMangement
	__host__ int* getPtr();			//return memory ptr;
	__host__ ~HostVariable();		//deallocate
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariable::HostVariable(int dm):
	DomainSize(dm),dbg(true){
	if(dbg)printf("\033[34mWarn\033[0m::HostVariable::constructor::ALLOCATION\n");
	cudaMalloc((void**)&dMem,sizeof(int)*DomainSize);
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariable::~HostVariable(){
	if(dbg)printf("\033[34mWarn\033[0m::HostVariable::destructor::DELLOCATION\n");
	cudaFree(dMem);
}

///////////////////////////////////////////////////////////////////////

__host__ int* HostVariable::getPtr()
	{return dMem;}


///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariable{
	int ground;			//track if variable is ground
	int changed;		//track if variable was modified
	int failed;			//track if variable is in a failed state
	int domainSize;		//size of the domain

	bool dbg;			//verbose

	int* domain;		//ptr to domain memory

	bool fullParallel;	//choose always parallel code execution 

	DeviceMemoryManagement deviceMemoryManagement;	
						//structure for fast modification
						//of the memory

	__device__ DeviceVariable();			//do nothing
	__device__ DeviceVariable(int*,int); 	//initialize
	__device__ void init(int*, int);		//initialize
	__device__ void init2(int*, int);		//initialize without setting
	__device__ ~DeviceVariable();			//do nothing

	__device__ int assign(int);			//assign choesen variable and returns 0.
										//otherwise -1
	__device__ int undoAssign(int);		//undo assignement
	__device__ void addTo(int,int);		//increment or decrement by delta

	__device__ void checkGround();		//check if variable is in ground state and modify ground
	__device__ void checkFailed();		//check if variable is in failed state and modify failed

	__device__ void print();			//stampa with modes

};

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::DeviceVariable(){}

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::~DeviceVariable(){}

///////////////////////////////////////////////////////////////////////

__device__ inline DeviceVariable::DeviceVariable(int* dMem, int ds):
	domainSize(ds),deviceMemoryManagement(dMem,1,1,domainSize),
	ground(-1),changed(-1),failed(-1),dbg(true),fullParallel(true),
	domain(dMem){
		if(fullParallel) deviceMemoryManagement.setFromToMulti(0,0,0,0,0,ds-1,1);
		else deviceMemoryManagement.setMatrix(0,1);
	}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::init(int* dMem, int ds){
	domainSize = ds;
	deviceMemoryManagement.init(dMem,1,1,ds);
	domain = dMem;
	fullParallel = true;
	ground  = -1;
	changed = -1;
	failed  = -1;
	dbg = true;

	if(fullParallel) deviceMemoryManagement.setFromToMulti(0,0,0,0,0,ds-1,1);
	else deviceMemoryManagement.setMatrix(0,1);
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::init2(int* dMem, int ds){
	domainSize = ds;
	deviceMemoryManagement.init(dMem,1,1,ds);
	domain = dMem;
	fullParallel = true;
	ground  = -1;
	changed = -1;
	failed  = -1;
	dbg = true;
}

///////////////////////////////////////////////////////////////////////


__device__ inline void externAssignSequential(int* domain, int size, int value){
	for(int i = 0; i < size; ++i){
		if(i != value)--domain[i];
	}
}

__global__ void externAssignParallel(int* domain, int size, int value){
	if(threadIdx.x + blockIdx.x * blockDim.x < size && 
	   threadIdx.x + blockIdx.x * blockDim.x != value)
		--domain[threadIdx.x + blockIdx.x * blockDim.x];
}

__device__ inline int DeviceVariable::assign(int value){
	if(value < 0 || value >= domainSize){
		printf("\033[31mError\033[0m::Variable::assign::ASSIGNMENT OUT OF BOUND\n");
		return -1;
	}

	if(failed == 1){
		printf("\033[31mError\033[0m::Variable::assign::VARIABLE ALREADY FAILED\n");
		return -1;
	}

	if(domain[value]<=0){
		printf("\033[31mError\033[0m::Variable::assign::VALUE NO MORE IN DOMAIN\n");
		return -1;
	}

	if(ground >= 0 && value != ground){
		printf("\033[31mError\033[0m::Variable::assign::VARIABLE NOT GROUND\n");
		return -1;
	}

	if(fullParallel)externAssignSequential(domain, domainSize, value);
	else{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		externAssignParallel<<<1,domainSize,0,s>>>(domain, domainSize, value);
		cudaStreamDestroy(s);
		cudaDeviceSynchronize();
	} 

	ground = value;
	return 0;

}		

///////////////////////////////////////////////////////////////////////

__device__ inline void externUndoAssignSequential(int* domain, int size, int value){
	for(int i = 0; i < size; ++i){
		if(i != value)++domain[i];
	}
}

__global__ void externUndoAssignParallel(int* domain, int size, int value){
	if(threadIdx.x + blockIdx.x * blockDim.x < size && 
	   threadIdx.x + blockIdx.x * blockDim.x != value)
		++domain[threadIdx.x + blockIdx.x * blockDim.x];
}

__device__ inline int DeviceVariable::undoAssign(int value){
	if(value < 0 || value >= domainSize){
		printf("\033[31mError\033[0m::Variable::undoAssign::OUT OF BOUND\n");
		return -1;
	}

	if(ground == -1){
		printf("\033[31mError\033[0m::Variable::undoAssign::VARIABLE NOT GROUND\n");
		return -1;
	}

	if(fullParallel)externUndoAssignSequential(domain, domainSize, value);
	else{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		externUndoAssignParallel<<<1,domainSize>>>(domain, domainSize, value);
		cudaStreamDestroy(s);
		cudaDeviceSynchronize();
	} 

	checkGround();

	return 0;

}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::addTo(int value, int delta){
	if(value < 0 || value >= domainSize){
		printf("\033[31mError\033[0m::Variable::addTo::ADDING OUT OF BOUND\n");
		return;
	}
	
	if(domain[value] > 0 && domain[value] + delta <= 0) changed = 1;

	domain[value]+=delta;

	checkGround();
	checkFailed();
	
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::checkGround(){
	int sum = 0;
	for(int i = 0; i < domainSize; ++i){
		if(domain[i]==1){
			++sum;
			ground = i;
		}
	}
	if(sum != 1) ground = -1;

}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::checkFailed(){
	for(int i = 0; i < domainSize; ++i)
		if(domain[i]==1){
			failed = -1;
			return;
		}
	failed = 1;
}

///////////////////////////////////////////////////////////////////////

__device__ inline void DeviceVariable::print(){
	for (int i = 0; i < domainSize; ++i){
		if(domain[i] == 0)
			printf("\033[31m%d\033[0m ", domain[i]);
		else if(domain[i] > 0)printf("\033[34m%d\033[0m ", domain[i]);
		else if(domain[i] < 0)printf("\033[31m%d\033[0m ", -domain[i]);
	}

	if(ground >= 0)printf(" ::: \033[32mgrd:%d\033[0m ", ground);
	else printf(" ::: grd:%d ", ground);

	if(changed == 1)printf("\033[31mchd:%d\033[0m ", changed);
	else printf("chd:%d ", changed);

	if(failed == 1)printf("\033[31mfld:%d\033[0m ", failed);
	else printf("fld:%d ", failed);

	printf("sz:%d\n", domainSize);
}

///////////////////////////////////////////////////////////////////////
