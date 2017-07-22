#pragma once
#include "../Variable/Variable2.cu"
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariableCollection{
	int* dMem;						//ptr to deviceMemory
	DeviceVariable* dMemVariables;	//vector for variables struct
	int nQueen;						//number of variables and also domain size
	bool dbg;						//verbose mode

	HostMemoryManagement hostMemoryManagement;

	__host__ HostVariableCollection(int);		//allocate memory with hostMemoryManagemnt
	__host__ ~HostVariableCollection();			//deallocate dMemVariables
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::HostVariableCollection(int nq):
	nQueen(nq),dbg(true),hostMemoryManagement(1,nQueen,nQueen){
	if(dbg)printf("\033[34mWarn\033[0m::HostVariableCollection::constructor::ALLOCATION\n");
	cudaMalloc((void**)&dMemVariables,sizeof(DeviceVariable)*nQueen);
	dMem = hostMemoryManagement.getPtr();
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::~HostVariableCollection(){
	if(dbg)printf("\033[34mWarn\033[0m::HostVariableCollection::destructor::DELLOCATION\n");
	cudaFree(dMemVariables);
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariableCollection{

	bool dbg;					//verbose
	bool fullParallel;			//chose parallel code

	int nQueen;					//number of variables and domain size
	DeviceVariable* variables;	//array for variables

	DeviceMemoryManagement deviceMemoryManagement;	
						//structure for fast modification
						//of the memory

	__device__ DeviceVariableCollection();							//do nothing
	__device__ DeviceVariableCollection(DeviceVariable*,int*,int);	//initialize
	__device__ void init(DeviceVariable*,int*,int);					//initialize
	__device__ ~DeviceVariableCollection();							//do nothing

	__device__ bool isGround();			//check if every variable is not failed
	__device__ bool isFailed();			//check if every variable is ground

	__device__ void print();			//print collection

};

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(DeviceVariable* dv, int* vm, int nq):
	dbg(true),fullParallel(true),nQueen(nq),variables(dv),deviceMemoryManagement(vm,1,nQueen,nQueen){
	
	if(fullParallel)deviceMemoryManagement.setMatrixFromToMultiLess(0,0,1);
	else deviceMemoryManagement.setMatrix(0,1);
	for (int i = 0; i < nQueen; ++i)
		variables[i].init2(vm,nQueen);
}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init(DeviceVariable* dv, int* vm, int nq){
	dbg = true;
	fullParallel = true;
	nQueen = nq;
	variables = dv;
	deviceMemoryManagement.init(vm,1,nQueen,nQueen);
	if(fullParallel)deviceMemoryManagement.setMatrixFromToMultiLess(0,0,1);
	else deviceMemoryManagement.setMatrix(0,1);
	for (int i = 0; i < nQueen; ++i)
		variables[i].init2(vm,nQueen);
}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::~DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::print(){
	for (int i = 0; i < nQueen; ++i){
		variables[i].print();
	}printf("\n");
}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isGround(){
	for(int i = 0; i < nQueen; ++i)
		if(variables[i].ground==-1)return false;

	return true;
}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isFailed(){
	for(int i = 0; i < nQueen; ++i)
		if(variables[i].failed == 1)return true;

	return false;
}

///////////////////////////////////////////////////////////////////////