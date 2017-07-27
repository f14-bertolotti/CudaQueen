#pragma once
#include "../Variable/Variable.cu"
#include "../TripleQueue/TripleQueue.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariableCollection{
	int* dMem;							//ptr to deviceMemory
	DeviceVariable* deviceVariableMem;	//vector for variables struct
	int* dMemlastValues;				//last values array
	int nQueen;							//number of variables and also domain size
	bool dbg;							//verbose mode
	HostQueue hostQueue;				//queue

	__host__ HostVariableCollection(int);		//allocate memory with hostMemoryManagemnt
	__host__ ~HostVariableCollection();			//deallocate dMemVariables
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::HostVariableCollection(int nq):
	nQueen(nq),dbg(true),hostQueue(nq){

	if(dbg)printf("\033[34mWarn\033[0m::HostVariableCollection::constructor::ALLOCATION\n");
	cudaMalloc((void**)&deviceVariableMem,sizeof(DeviceVariable)*nQueen);
	cudaMalloc((void**)&dMemlastValues,sizeof(int)*nQueen);
	cudaMalloc((void**)&dMem,sizeof(int)*nQueen*nQueen);
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::~HostVariableCollection(){
	if(dbg)printf("\033[34mWarn\033[0m::HostVariableCollection::destructor::DELLOCATION\n");
	cudaFree(deviceVariableMem);
	cudaFree(dMemlastValues);
	cudaFree(dMem);
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariableCollection{

	bool dbg;						//verbose
	bool fullParallel;				//chose parallel code
	int nQueen;						//number of variables and domain size
	int* lastValues;				//last values array
	int* dMem;	
	DeviceVariable* deviceVariable;	//array for variables
	DeviceQueue deviceQueue;		//triple queue

	__device__ DeviceVariableCollection();											//do nothing
	__device__ DeviceVariableCollection(DeviceVariable*,Triple*, int*,int*,int);	//initialize
	__device__ void init(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ void init2(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ ~DeviceVariableCollection();											//do nothing

	__device__ bool isGround();			//check if every variable is not failed
	__device__ bool isFailed();			//check if every variable is ground

	__device__ void print();			//print collection

};

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq):
	dbg(true),fullParallel(true),nQueen(nq),deviceVariable(dv),deviceQueue(q,nq),lastValues(lv),dMem(vm){
	
	for(int i = 0; i < nQueen*nQueen; ++i){
		vm[i] = 1;
	}

	for (int i = 0; i < nQueen; ++i){
		deviceVariable[i].init2(&vm[nQueen*i],nQueen);
		lastValues[i]=0;
	}
}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){
	dbg = true;
	dMem = vm;
	fullParallel = true;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

	for(int i = 0; i < nQueen*nQueen; ++i){
		vm[i] = 1;
	}

	for (int i = 0; i < nQueen; ++i){
		deviceVariable[i].init2(&vm[nQueen*i],nQueen);
		lastValues[i]=0;
	}
}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::init2(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){
	dbg = true;
	fullParallel = true;
	dMem = vm;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

	for (int i = 0; i < nQueen; ++i){
		deviceVariable[i].init2(&vm[nQueen*i],nQueen);
		lastValues[i]=0;
	}
}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::~DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceVariableCollection::print(){
	for (int i = 0; i < nQueen; ++i){
		printf("[%d] ::: ",lastValues[i]);
		deviceVariable[i].print();
	}
	deviceQueue.print();
	printf("\n");
}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isGround(){
	for(int i = 0; i < nQueen; ++i)
		if(deviceVariable[i].ground==-1)return false;

	return true;
}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceVariableCollection::isFailed(){
	for(int i = 0; i < nQueen; ++i)
		if(deviceVariable[i].failed == 1)return true;

	return false;
}

///////////////////////////////////////////////////////////////////////
