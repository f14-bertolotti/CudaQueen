#pragma once
#include "../Variable/Variable.cu"
#include "../TripleQueue/TripleQueue.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenPropagation/QueenPropagation.cu"
#include "../QueenConstraints/QueenConstraints.cu"
#include "../ErrorChecking/ErrorChecking.cu"
#include <cstdio>

///////////////////////////////////////////////////////////////////////
////////////////////////HOST SIDE//////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct HostParallelQueue{

	int* lockReading;
	DeviceVariableCollection* deviceVariableCollection;

	DeviceVariable* deviceVariable;
	int* variablesMem;
	int* lastValuesMem;
	Triple* tripleQueueMem;

	int size;
	int nQueen;

	__host__ HostParallelQueue(int,int);
	__host__ ~HostParallelQueue();
};

//////////////////////////////////////////////////////////////////////////////////////////////

__host__ HostParallelQueue::HostParallelQueue(int nq, int sz):nQueen(nq),size(sz){
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&lockReading,sizeof(int)*size),"Error::HostParallelQueue::ALLOCATE 1");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariableCollection,sizeof(DeviceVariableCollection)*size),"Error::HostParallelQueue::ALLOCATE 2");

	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariable,sizeof(DeviceVariable)*size*nQueen),"HostParallelQueue::DEVICE VARIABLE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&variablesMem,sizeof(int)*nQueen*nQueen*size),"HostParallelQueue::VARIABLE MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&lastValuesMem,sizeof(int)*nQueen*size),"HostParallelQueue::LAST VALUES MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&tripleQueueMem,sizeof(Triple)*nQueen*nQueen*3*size),"HostParallelQueue::TRIPLE QUEUE MEM ALLOCATION");

	externSet<<<int(size*nQueen*nQueen)/1000+1,1000>>>(variablesMem,lastValuesMem,nQueen,size);
	ErrorChecking::hostErrorCheck(cudaPeekAtLastError(),"HostParallelQueue::EXTERN SET CALL");
	ErrorChecking::hostErrorCheck(cudaDeviceSynchronize(),"HostParallelQueue::SYNCH");

}

//////////////////////////////////////////////////////////////////////////////////////////////

__host__ HostParallelQueue::~HostParallelQueue(){
	ErrorChecking::hostErrorCheck(cudaFree(lockReading),"Error::hostParallelQueue::DEALLOCATE 1");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableCollection),"Error::hostParallelQueue::DEALLOCATE 2");

	ErrorChecking::hostErrorCheck(cudaFree(variablesMem),"Error::hostParallelQueue::VARIABLES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(lastValuesMem),"Error::hostParallelQueue::LAST VALUES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(tripleQueueMem),"Error::hostParallelQueue::TRIPLE QUEUE ME DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableCollection),"Error::hostParallelQueue::DEVICE VARIABLE COLLECTION DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariable),"Error::hostParallelQueue::DEVICE VARIABLE DEALLOCATION");
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceParallelQueue{
	int lockCount;								//lock on count variable
	int count;									//number of element in queue
	int size;									//max number of element(fixed)
	int nQueen;

	int* lockReading;
	DeviceVariableCollection* deviceVariableCollection;
	DeviceVariable* deviceVariable;
	int* variablesMem;
	int* lastValuesMem;
	Triple* tripleQueueMem;

	__device__ DeviceParallelQueue();					//do nothing
	__device__ DeviceParallelQueue(DeviceVariableCollection*,DeviceVariable*,int*,int*,int*,Triple*,int,int);	//initialize
	__device__ void init(DeviceVariableCollection*,DeviceVariable*,int*,int*,int*,Triple*,int,int);				//initialize

	__device__ int add(DeviceVariableCollection&);						//add an element, -1 if fail
	__device__ int pop();												//delete last element , -1 if fail
	__device__ int frontAndPop(DeviceVariableCollection&);				//returns last and delete last element, -1 if fail
	__device__ bool empty();											//return true if empty

	__device__ void print();					//print
	__device__ void printLocks();				//do nothing
												//prints are not locked

	__device__ ~DeviceParallelQueue();				//do nothing
};

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceParallelQueue::DeviceParallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceParallelQueue::DeviceParallelQueue(DeviceVariableCollection* dvc, 
													DeviceVariable* dv,
													int* vm, int* lvm, int* lr, 
													Triple* tqm,
													int nq, int sz):
													deviceVariableCollection(dvc),deviceVariable(dv),
													variablesMem(vm),lastValuesMem(lvm),lockReading(lr),tripleQueueMem(tqm),
													nQueen(nq),size(sz),
													count(0),lockCount(0){}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::init(DeviceVariableCollection* dvc, DeviceVariable* dv,
									 	  int* vm, int* lvm, int* lr, Triple* tqm, int nq, int sz){

	variablesMem = vm;
	lastValuesMem = lvm;
	tripleQueueMem = tqm;

	deviceVariable = dv;
	deviceVariableCollection = dvc;

	nQueen = nq;
	size = sz;

	count = 0;
	lockCount = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::add(DeviceVariableCollection& element){

	int temp = 0;
	while(atomicCAS(&lockCount,0,1)==1){}
	if(count == size){
		ErrorChecking::deviceMessage("Warn::DeviceParallelQueue::add::NOT ENOUGH SPACE");
		lockCount = 0;
		return -1;
	}
	temp = count;
	++count;
	lockCount = 0;

	while(atomicCAS(&lockReading[temp],0,1)==1){}
	deviceVariableCollection[temp] = element;
	lockReading[temp] = 0;

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::frontAndPop(DeviceVariableCollection& element){

	while(atomicCAS(&lockCount,0,1)==1){}
	if(count <= 0){
		ErrorChecking::deviceMessage("Warn::DeviceParallelQueue::frontAndPop::OUT OF BOUNDS");
		lockCount = 0;
		return -1;
	}
	while(atomicCAS(&lockReading[count-1],0,1)==1){}

	count--;

	element = deviceVariableCollection[count];
	lockReading[count] = 0;
	lockCount = 0;

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::pop(){
	while(atomicCAS(&lockCount,0,1)==1){}
	if(count <= 0){
		ErrorChecking::deviceMessage("Warn::DeviceParallelQueue::pop::EMPTY QUEUE");
		lockCount = 0;
		return -1;
	}
	while(atomicCAS(&(lockReading[count-1]),0,1)==1){}
	--count;
	lockReading[count] = 0;
	lockCount = 0;

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool DeviceParallelQueue::empty(){
	int temp = 0;
	while(atomicCAS(&lockCount,0,1)==1){}
	temp = count;
	lockCount = 0;
	return temp==0 ? true : false;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::print(){

	for(int i = 0; i < count; ++i) {
		printf("index: %d - ", i);
		deviceVariableCollection[i].print();
	}

	printf("count:%d\n",count);
	printf("size:%d\n",size);
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::printLocks(){

	printf("lock count: %d\n",lockCount);
	printf("locks reading:\n");
	for (int i = 0; i < size; ++i){
		if(i%100==0 && i != 0)printf("\n");
		printf("%d",lockReading[i]);
	}
	printf("\n");
}


//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceParallelQueue::~DeviceParallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////
