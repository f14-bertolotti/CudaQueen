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

	DeviceVariableCollection* deviceVariableCollection;

	DeviceVariable* deviceVariable;
	int* variablesMem;
	int* lastValuesMem;
	int* lockReading;
	int* levelLeaved;
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
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&levelLeaved,sizeof(int)*size),"Error::HostParallelQueue::ALLOCATE 3");

	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariable,sizeof(DeviceVariable)*size*nQueen),"HostParallelQueue::DEVICE VARIABLE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&variablesMem,sizeof(int)*nQueen*nQueen*size),"HostParallelQueue::VARIABLE MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&lastValuesMem,sizeof(int)*nQueen*size),"HostParallelQueue::LAST VALUES MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&tripleQueueMem,sizeof(Triple)*nQueen*nQueen*3*size),"HostParallelQueue::TRIPLE QUEUE MEM ALLOCATION");

	ErrorChecking::hostErrorCheck(cudaPeekAtLastError(),"HostParallelQueue::EXTERN SET CALL");
	ErrorChecking::hostErrorCheck(cudaDeviceSynchronize(),"HostParallelQueue::SYNCH");

}

//////////////////////////////////////////////////////////////////////////////////////////////

__host__ HostParallelQueue::~HostParallelQueue(){
	ErrorChecking::hostErrorCheck(cudaFree(levelLeaved),"Error::hostParallelQueue::DEALLOCATE 1");
	ErrorChecking::hostErrorCheck(cudaFree(lockReading),"Error::hostParallelQueue::DEALLOCATE 1");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableCollection),"Error::hostParallelQueue::DEALLOCATE 2");
	ErrorChecking::hostErrorCheck(cudaFree(variablesMem),"Error::hostParallelQueue::VARIABLES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(lastValuesMem),"Error::hostParallelQueue::LAST VALUES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(tripleQueueMem),"Error::hostParallelQueue::TRIPLE QUEUE ME DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariable),"Error::hostParallelQueue::DEVICE VARIABLE DEALLOCATION");
}

///////////////////////////////////////////////////////////////////////
////////////////////////DEVICE SIDE////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct DeviceParallelQueue{
	int size;									//max number of element(fixed)
	int nQueen;									//size of csp
	int maxUsed;
	int lockMaxUsed;

	DeviceVariableCollection* deviceVariableCollection;
	DeviceVariable* deviceVariable;
	int* lockReading;
	int* variablesMem;
	int* lastValuesMem;
	int* levelLeaved;
	Triple* tripleQueueMem;

	__device__ DeviceParallelQueue();					//do nothing
	__device__ DeviceParallelQueue(DeviceVariableCollection*,DeviceVariable*,int*,int*,int*,int*,Triple*,int,int);	//initialize
	__device__ void init(DeviceVariableCollection*,DeviceVariable*,int*,int*,int*,int*,Triple*,int,int);			//initialize

	__device__ int add(DeviceVariableCollection&,int,int);		//add an element, -1 if fail
	__device__ int read(DeviceVariableCollection&,int);			//returns last and delete last element, -1 if fail
	__device__ int expansion(DeviceVariableCollection&, int);	//expansion as WorkSet

	__device__ void print();					//print
	__device__ void printLocks();
	__device__ int stillInQueue();

	__device__ ~DeviceParallelQueue();				//do nothing
};

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceParallelQueue::DeviceParallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void ParallelQueueExternInit(DeviceVariableCollection* deviceVariableCollection,
									    DeviceVariable* deviceVariable, int* variablesMem,
									    int* lastValuesMem, int* lockReading, Triple* tripleQueueMem,
									    int nQueen, int nVariableCollection){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < nQueen*nVariableCollection){

		deviceVariable[index].init2(&variablesMem[index*nQueen],nQueen);

		if(index < nVariableCollection){

			deviceVariableCollection[index].init2(&deviceVariable[index*nQueen],
												 &tripleQueueMem[index*nQueen*nQueen*3],
												 &variablesMem[index*nQueen*nQueen],
												 &lastValuesMem[index*nQueen],nQueen);

		}

	}

	if(index < nVariableCollection)
		lockReading[index] = 0;
}

__device__ DeviceParallelQueue::DeviceParallelQueue(DeviceVariableCollection* dvc, 
													DeviceVariable* dv,
													int* vm, int* lvm, int* lr, int* ll,
													Triple* tqm,
													int nq, int sz):
													deviceVariableCollection(dvc),deviceVariable(dv),
													variablesMem(vm),levelLeaved(ll),lastValuesMem(lvm),tripleQueueMem(tqm),
													lockReading(lr),nQueen(nq),size(sz),maxUsed(0),lockMaxUsed(0){

	ParallelQueueExternInit<<<int(size*nQueen)/1000+1,1000>>>(deviceVariableCollection,
															  deviceVariable,
															  variablesMem,
											    			  lastValuesMem,
											    			  lockReading,
															  tripleQueueMem,
															  nQueen,size);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceParallelQueue::DeviceParallelQueue::EXTERN INIT");


}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::init(DeviceVariableCollection* dvc, DeviceVariable* dv,
								 	  int* vm, int* lvm, int* lr, int* ll, Triple* tqm, int nq, int sz){

	variablesMem = vm;
	lastValuesMem = lvm;
	tripleQueueMem = tqm;

	deviceVariable = dv;
	deviceVariableCollection = dvc;

	lockReading = lr;
	levelLeaved = ll;

	nQueen = nq;
	size = sz;
	maxUsed = 0;
	lockMaxUsed = 0;

	ParallelQueueExternInit<<<int(size*nQueen)/1000+1,1000>>>(deviceVariableCollection,
											 				  deviceVariable,
											 				  variablesMem,
							    			 				  lastValuesMem,
							    			 				  lockReading,
											 				  tripleQueueMem,
											 				  nQueen,size);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceParallelQueue::init::EXTERN INIT");
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::add(DeviceVariableCollection& element, int level, int index){

	int pos = -1;
	for (int i = 0; i < size; ++i){
		if(atomicCAS(&lockReading[i],0,1)==0){
			pos = i;
			break;
		}
	}

	if(pos == -1)return -1;

	while(atomicCAS(&lockMaxUsed,0,1)==1){}
	if(pos >= maxUsed)maxUsed = pos+1;
	lockMaxUsed = 0;

	levelLeaved[pos] = level;
	deviceVariableCollection[pos] = element;

	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"SYNCH");

	lockReading[pos] = 2;

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::read(DeviceVariableCollection& element, int index){

	int pos = -1;
	for (int i = 0; i < maxUsed && i < size; ++i){
		if(atomicCAS(&lockReading[i],2,3)==2){
			pos = i;
			break;
		}
	}

	if(pos == -1)return -1;
	element = deviceVariableCollection[pos];

	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"SYNCH");

	int ltemp = levelLeaved[pos];

	lockReading[pos] = 0;

	return ltemp;
}


//////////////////////////////////////////////////////////////////////////////////////////////


__global__ void externCopy(DeviceVariableCollection& toCopy, DeviceVariableCollection& in, int level){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int nQueen = toCopy.nQueen;

	if(index < nQueen*nQueen){
		in.dMem[index] = toCopy.dMem[index];
	}

	if(index < toCopy.nQueen*nQueen){
		in.deviceQueue.q[index] = toCopy.deviceQueue.q[index];
		in.deviceQueue.q[nQueen*nQueen+index] = toCopy.deviceQueue.q[nQueen*nQueen+index];
		in.deviceQueue.q[2*nQueen*nQueen+index] = toCopy.deviceQueue.q[2*nQueen*nQueen+index];
	}

	if(index < nQueen){
		if(index != level)in.deviceVariable[index].ground = toCopy.deviceVariable[index].ground;
		in.deviceVariable[index].failed = toCopy.deviceVariable[index].failed;
		in.deviceVariable[index].changed = toCopy.deviceVariable[index].changed;
		in.deviceVariable[index].domainSize = toCopy.deviceVariable[index].domainSize;	
		in.lastValues[index] = toCopy.lastValues[index];
	}

	if(index == 1){
		in.deviceQueue.count = toCopy.deviceQueue.count;
		in.lastValues[level] = nQueen;
	}

}

__global__ void end(int* lock){
	*lock = 2;
}

__device__ int DeviceParallelQueue::expansion(DeviceVariableCollection& element, int level){


	DeviceQueenPropagation deviceQueenPropagation;
	if(nQueen > 100) return -1;
	int first = -1;
	int nValues = 0;
	int positions[100];
	int values[100];
	bool ok = false;

	for(int val = 0; val < nQueen; ++val){

		if(element.deviceVariable[level].domain[val] == 1 && first != -1){

			ok = false;

			for (int i = 0; i < size; ++i){

				if(atomicCAS(&lockReading[i],0,1)==0){

					positions[nValues] = i;
					values[nValues] = val;
					ok = true;
					++nValues;
					break;

				}

			}

			if(!ok){

				return -1;

			}
		}else if(element.deviceVariable[level].domain[val] == 1 && first == -1){

			first = val;

		}
	}


	for(int i = 0; i < nValues; ++i){

		deviceVariableCollection[positions[i]].deviceVariable[level].ground = values[i];

		while(atomicCAS(&lockMaxUsed,0,1)==1){}
		if(positions[i] >= maxUsed)maxUsed = positions[i]+1;
		lockMaxUsed = 0;

		levelLeaved[positions[i]] = level+1;

		cudaStream_t s;
		ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceParallelQueue::expansion");

		externCopy<<<1,nQueen*nQueen,0,s>>>(element,deviceVariableCollection[positions[i]],level);
		externAssignParallel<<<1,deviceVariableCollection[positions[i]].deviceVariable[level].domainSize,0,s>>>(
													deviceVariableCollection[positions[i]].deviceVariable[level].domain, 
													deviceVariableCollection[positions[i]].deviceVariable[level].domainSize, values[i]
													);
		deviceVariableCollection[positions[i]].deviceVariable[level].ground = values[i];
		deviceQueenPropagation.parallelForwardPropagation2(deviceVariableCollection[positions[i]],level,values[i],s);
		end<<<1,1,0,s>>>(&lockReading[positions[i]]);
		ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceParallelQueue::EXPANDED");
		ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceParallelQueue::expansion::STREAM DESTRUCTION");


	}


	int val = first;

	element.lastValues[level] = nQueen;
	element.deviceVariable[level].ground = val;

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceParallelQueue::expansion");

	externAssignParallel<<<1,element.deviceVariable[level].domainSize,0,s>>>(
												element.deviceVariable[level].domain, 
												element.deviceVariable[level].domainSize, val
												);
	deviceQueenPropagation.parallelForwardPropagation2(element,level,val,s);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceParallelQueue::EXPANDED");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceParallelQueue::expansion::STREAM DESTRUCTION");

	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceParallelQueue::SYNCH");

	return nValues+1;

}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::print(){

	int count = 0;
	for(int i = 0; i < size; ++i) {
		if(lockReading[i] != 0)printf("------[%d,%d,%d]------\n", i,lockReading[i],levelLeaved[i]);
		if(lockReading[i] != 0)deviceVariableCollection[i].print();
		if(lockReading[i] != 0)++count;
	}

	printf("count:%d \n",count);
	printf("size: %d\n",size);
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceParallelQueue::printLocks(){
	for(int i = 0; i < size; ++i){
		if(i % 100 == 0)printf("\n");
		printf("%d", lockReading[i]);
	}printf("\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceParallelQueue::stillInQueue(){
	int sum = 0;
	for(int i = 0; i < size; ++i){
		if(lockReading[i] > 0)++sum;
	}
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceParallelQueue::~DeviceParallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////
