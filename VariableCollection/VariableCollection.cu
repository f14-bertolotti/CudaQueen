#pragma once
#include "../Variable/Variable.cu"
#include "../TripleQueue/TripleQueue.cu"
#include "../ErrorChecking/ErrorChecking.cu"
#include "../MemoryManagement/MemoryManagement.cu"

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostVariableCollection{
	int* dMem;							//ptr to deviceMemory
	DeviceVariable* deviceVariableMem;	//vector for variables struct
	int* dMemlastValues;				//last values array
	int nQueen;							//number of variables and also domain size
	HostQueue hostQueue;				//queue

	__host__ HostVariableCollection(int);		//allocate memory with hostMemoryManagemnt
	__host__ ~HostVariableCollection();			//deallocate dMemVariables
};

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::HostVariableCollection(int nq):
	nQueen(nq),hostQueue(nq){

	ErrorChecking::hostMessage("Warn::HostVariableCollection::constructor::ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariableMem,sizeof(DeviceVariable)*nQueen),"HostVariableCollection::HostVariableCollection::DEVICE VARIABLE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMemlastValues,sizeof(int)*nQueen),"HostVariableCollection::HostVariableCollection::LAST VALUE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&dMem,sizeof(int)*nQueen*nQueen),"HostVariableCollection::HostVariableCollection::VARIABLE MEM ALLOCATION");
}

///////////////////////////////////////////////////////////////////////

__host__ HostVariableCollection::~HostVariableCollection(){
	ErrorChecking::hostMessage("Warn::HostVariableCollection::destructor::DELLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableMem),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
	ErrorChecking::hostErrorCheck(cudaFree(dMemlastValues),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
	ErrorChecking::hostErrorCheck(cudaFree(dMem),"HostVariableCollection::~HostVariableCollection::DEVICE VARIABLE DEALLOCATION");;
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceVariableCollection{

	int fullParallel;				//chose parallel code
	int nQueen;						//number of variables and domain size
	int* lastValues;				//last values array
	int* dMem;	
	DeviceVariable* deviceVariable;	//array for variables
	DeviceQueue deviceQueue;		//triple queue

	__device__ DeviceVariableCollection();											//do nothing
	__device__ DeviceVariableCollection(DeviceVariable*,Triple*, int*,int*,int);	//initialize
	__device__ void init(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ void init2(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ void init3(DeviceVariable*,Triple*,int*,int*,int);					//initialize
	__device__ ~DeviceVariableCollection();											//do nothing

	__device__ DeviceVariableCollection& operator=(const DeviceVariableCollection&);			//copy

	__device__ bool isGround();			//check if every variable is not failed
	__device__ bool isFailed();			//check if every variable is ground

	__device__ void print();			//print collection

};

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(){}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::DeviceVariableCollection(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq):
	fullParallel(true),nQueen(nq),deviceVariable(dv),deviceQueue(q,nq),lastValues(lv),dMem(vm){
	
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

__device__ void DeviceVariableCollection::init3(DeviceVariable* dv,Triple* q, int* vm, int* lv, int nq){

	fullParallel = true;
	dMem = vm;
	nQueen = nq;
	deviceVariable = dv;
	lastValues = lv;
	deviceQueue.init(q,nq);

}

///////////////////////////////////////////////////////////////////////

__device__ DeviceVariableCollection::~DeviceVariableCollection(){}

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

__device__ DeviceVariableCollection& DeviceVariableCollection::operator=(const DeviceVariableCollection& other){

	MemoryManagement<Triple>::copy(other.deviceQueue.q, deviceQueue.q, nQueen*nQueen*3);
	MemoryManagement<int>::copy(other.lastValues, lastValues, nQueen);
	MemoryManagement<int>::copy(other.dMem, dMem, nQueen*nQueen);

	deviceQueue.count = other.deviceQueue.count;

	for(int i = 0; i < nQueen; ++i){
		deviceVariable[i].ground = other.deviceVariable[i].ground;
		deviceVariable[i].failed = other.deviceVariable[i].failed;
		deviceVariable[i].changed = other.deviceVariable[i].changed;
		deviceVariable[i].domainSize = other.deviceVariable[i].domainSize;		
	}

	return *this;
}

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