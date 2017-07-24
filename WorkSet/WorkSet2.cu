#pragma once
#include "../Variable/Variable2.cu"
#include "../MemoryManagement/MemoryManagement.cu"
#include "../TripleQueue/TripleQueue2.cu"
#include "../VariableCollection/VariableCollection2.cu"
#include "../QueenPropagation/QueenPropagation2.cu"
#include <cstdio>

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////


struct HostWorkSet{

	DeviceVariableCollection* deviceVariableCollection;
	DeviceVariable* deviceVariable;
	int* variablesMem;
	int* lastValuesMem;
	Triple* tripleQueueMem;
	int nQueen;
	int nVariableCollection;

	__host__ HostWorkSet(int,int);	//allocate memory for queue, lastValues, and all variables
									//do this for nVariableColletion matrix of nQueen as row size
									//and column size 
	__host__ ~HostWorkSet();		//free all memory

};

///////////////////////////////////////////////////////////////////////

__global__ void externSet(int* variablesMem,int* lastValuesMem, int nQueen,int nVariableCollection){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < nVariableCollection*nQueen*nQueen){
		variablesMem[index] = 1;
		if(index < nVariableCollection*nQueen)
			lastValuesMem[index] = 0;
	}
}

__host__ HostWorkSet::HostWorkSet(int nq, int nvc):nQueen(nq),nVariableCollection(nvc){
	cudaMalloc((void**)&deviceVariableCollection,sizeof(DeviceVariableCollection)*nVariableCollection);
	cudaMalloc((void**)&deviceVariable,sizeof(DeviceVariable)*nVariableCollection*nQueen);
	cudaMalloc((void**)&variablesMem,sizeof(int)*nQueen*nQueen*nVariableCollection);
	cudaMalloc((void**)&lastValuesMem,sizeof(int)*nQueen*nVariableCollection);
	cudaMalloc((void**)&tripleQueueMem,sizeof(Triple)*nQueen*nQueen*3*nVariableCollection);

	externSet<<<int(nVariableCollection*nQueen*nQueen)/1000+1,1000>>>(variablesMem,lastValuesMem,nQueen,nVariableCollection);
}

///////////////////////////////////////////////////////////////////////

__host__ HostWorkSet::~HostWorkSet(){
	cudaFree(variablesMem);
	cudaFree(lastValuesMem);
	cudaFree(tripleQueueMem);
	cudaFree(deviceVariableCollection);
	cudaFree(deviceVariable);
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceWorkSet{

	int* variablesMem;
	int* lastValuesMem;
	Triple* tripleQueueMem;

	DeviceVariable* deviceVariable;
	DeviceVariableCollection* deviceVariableCollection;

	int nQueen;					//size of row and size of column
	int nVariableCollection;	//numbert of variable collection

	int count;					//currently variable collection used
	int lockCount;				//lock for count 

	int dbg;					//verbose mode
	int fullParallel;			//choose parallel code

	__device__ DeviceWorkSet();																		//do nothing
	__device__ DeviceWorkSet(DeviceVariableCollection*,DeviceVariable*,int*,int*,Triple*,int,int);	//initialize
	__device__ void init(DeviceVariableCollection*,DeviceVariable*,int*,int*,Triple*,int,int);		//initialize
	__device__ ~DeviceWorkSet();																	//do nothing

	__device__ int expand(int,int);	//prepare for parallel computation on a specific level
									//for a chosen variable collection, return number of expansions
									//-1 otherwise

	__device__ void print();

};

///////////////////////////////////////////////////////////////////////

__device__ DeviceWorkSet::DeviceWorkSet(){}

///////////////////////////////////////////////////////////////////////

__global__ void externInit(DeviceVariableCollection* deviceVariableCollection,
						   DeviceVariable* deviceVariable, int* variablesMem,
						   int* lastValuesMem, Triple* tripleQueueMem, int nQueen,
						   int nVariableCollection){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < nQueen*nVariableCollection){
		deviceVariable[index].init2(&variablesMem[index*nQueen],nQueen);
		if(index < nVariableCollection)
			deviceVariableCollection[index].init2(&deviceVariable[index*nQueen],
												 &tripleQueueMem[index*nQueen*nQueen*3],
												 &variablesMem[index*nQueen*nQueen],
												 &lastValuesMem[index*nQueen],nQueen);
	}

}

__device__ DeviceWorkSet::DeviceWorkSet(DeviceVariableCollection* dvc, DeviceVariable* dv,
										int* vm, int* lvm, Triple* tqm, int nq, int nvc):
										deviceVariableCollection(dvc),deviceVariable(dv),
										variablesMem(vm),lastValuesMem(lvm),tripleQueueMem(tqm),
										nQueen(nq),nVariableCollection(nvc),dbg(true),
										fullParallel(true),count(1),lockCount(0){

	if(fullParallel){
		externInit<<<int(nVariableCollection*nQueen)/1000+1,1000>>>(deviceVariableCollection,
																	deviceVariable,
																	variablesMem,
													    			lastValuesMem,
																	tripleQueueMem,
																	nQueen,nVariableCollection);
	}else{
		for(int i = 0; i < nVariableCollection*nQueen; ++i){
			deviceVariable[i].init2(&variablesMem[i*nQueen],nQueen);
			if(i < nVariableCollection)
				deviceVariableCollection[i].init2(&deviceVariable[i*nQueen],
												 &tripleQueueMem[i*nQueen*nQueen*3],
												 &variablesMem[i*nQueen*nQueen],
												 &lastValuesMem[i*nQueen],nQueen);
		}
	}

}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceWorkSet::init(DeviceVariableCollection* dvc, DeviceVariable* dv,
									int* vm, int* lvm, Triple* tqm, int nq, int nvc){

	variablesMem = vm;
	lastValuesMem = lvm;
	tripleQueueMem = tqm;

	deviceVariable = dv;
	deviceVariableCollection = dvc;

	nQueen = nq;
	nVariableCollection = nvc;

	dbg = true;
	fullParallel = true;

	count = 1;
	lockCount = 0;

	if(fullParallel){
		externInit<<<int(nVariableCollection*nQueen)/1000+1,1000>>>(deviceVariableCollection,
																	deviceVariable,
																	variablesMem,
													    			lastValuesMem,
																	tripleQueueMem,
																	nQueen,nVariableCollection);
	}else{
		for(int i = 0; i < nVariableCollection*nQueen; ++i){
			deviceVariable[i].init2(&variablesMem[i*nQueen],nQueen);
			if(i < nVariableCollection)
				deviceVariableCollection[i].init2(&deviceVariable[i*nQueen],
												 &tripleQueueMem[i*nQueen*nQueen*3],
												 &variablesMem[i*nQueen*nQueen],
												 &lastValuesMem[i*nQueen],nQueen);
		}
	}
}

///////////////////////////////////////////////////////////////////////

__device__ DeviceWorkSet::~DeviceWorkSet(){}

///////////////////////////////////////////////////////////////////////

__device__ void DeviceWorkSet::print(){

	for(int i = 0; i < nVariableCollection; ++i)
		deviceVariableCollection[i].print();

}

///////////////////////////////////////////////////////////////////////

__global__ void externExpand(DeviceWorkSet& deviceWorkSet, int who, int level, int nValues, int nQueen){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	DeviceQueenPropagation deviceQueenPropagation;

	if(index < nQueen*nQueen*3*nValues){
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*(who+1)+index].var = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].var;
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*(who+1)+index].val = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].val;
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*(who+1)+index].cs = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].cs;
	}

	if(index < nQueen*nQueen*nValues)
		deviceWorkSet.variablesMem[nQueen*nQueen*(who+1)+index] = 
			deviceWorkSet.variablesMem[nQueen*nQueen*who+(index%(nQueen*nQueen))];

	if(index < nQueen * nValues)
		deviceWorkSet.lastValuesMem[nQueen*(who+1)+index] = 
			deviceWorkSet.variablesMem[nQueen*who+(index%(nQueen))];

	if(index < nValues)
		deviceWorkSet.deviceVariableCollection[who+index+1].deviceQueue.count =
			deviceWorkSet.deviceVariableCollection[who].deviceQueue.count;

	__syncthreads();

	int j = 0;
	if(index == 1)
		for(int i = 0; i < nQueen; ++i){
			if(deviceWorkSet.deviceVariableCollection[who+1+j].variables[level].domain[i] == 1){
				cudaStream_t s;
				cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
				externAssignParallel<<<1,nQueen,0,s>>>(deviceWorkSet.deviceVariableCollection[who+1+j].variables[level].domain,nQueen,i);
				deviceWorkSet.deviceVariableCollection[who+1+j].variables[level].ground = i;
				cudaStreamDestroy(s);
				++j;
			}
		}

	__syncthreads();

	if(index < nValues)
		deviceQueenPropagation.parallelForwardPropagation(
			deviceWorkSet.deviceVariableCollection[who+index+1],
			level,
			deviceWorkSet.deviceVariableCollection[who+index+1].variables[level].ground);
}

__device__ int DeviceWorkSet::expand(int who, int level){

	if(who < 0 || who > nVariableCollection){
		printf("\033[31mError\033[0m::DeviceWorkSet::expand::VARIABLE COLLECTION INDEX OUT OF BOUND\n");
		return -1;
	}

	if(level < 0 || level > nQueen){
		printf("\033[31mError\033[0m::DeviceWorkSet::expand::LEVEL OUT OF BOUND\n");
		return -1;
	}

	int nValues = 0;
	for(int value = 0; value < nQueen; ++value)
		if(deviceVariableCollection[who].variables[level].domain[value] == 1)
			++nValues;

	if(nValues + count > nVariableCollection){
		printf("\033[31mError\033[0m::DeviceWorkSet::expand::NOT ENOUGH SPACE\n");
		return -1;
	}

	if(nValues == 0){
		printf("\033[34mWarn\033[0m::DeviceWorkSet::expand::VARIABLE IS FAILED\n");
		return 0;
	}

	while(atomicCAS(&lockCount,0,1)==1){}
	count+=nValues;
	lockCount = 0;

	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	externExpand<<<int(nQueen*nQueen*3*nValues)/1000+1,1000,0,s>>>(*this,who,level,nValues,nQueen);
	cudaStreamDestroy(s);
	cudaDeviceSynchronize();

	return nValues;
}




