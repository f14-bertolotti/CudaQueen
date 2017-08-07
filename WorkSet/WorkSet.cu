#pragma once
#include "../Variable/Variable.cu"
#include "../TripleQueue/TripleQueue.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../QueenPropagation/QueenPropagation.cu"
#include "../QueenConstraints/QueenConstraints.cu"
#include "../ErrorChecking/ErrorChecking.cu"
#include "../parallelQueue/parallelQueue.cu"
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void externSet(int* variablesMem,int* lastValuesMem, int nQueen,int nVariableCollection){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < nVariableCollection*nQueen*nQueen){
		variablesMem[index] = 1;
		if(index < nVariableCollection*nQueen)
			lastValuesMem[index] = 0;
	}

}

__host__ HostWorkSet::HostWorkSet(int nq, int nvc):nQueen(nq),nVariableCollection(nvc){

	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariableCollection,sizeof(DeviceVariableCollection)*nVariableCollection),"HostWorkSet::HostWorkSet::DEVICE VARIABLE COLLECTION ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&deviceVariable,sizeof(DeviceVariable)*nVariableCollection*nQueen),"HostWorkSet::HostWorkSet::DEVICE VARIABLE ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&variablesMem,sizeof(int)*nQueen*nQueen*nVariableCollection),"HostWorkSet::HostWorkSet::VARIABLE MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&lastValuesMem,sizeof(int)*nQueen*nVariableCollection),"HostWorkSet::HostWorkSet::LAST VALUES MEM ALLOCATION");
	ErrorChecking::hostErrorCheck(cudaMalloc((void**)&tripleQueueMem,sizeof(Triple)*nQueen*nQueen*3*nVariableCollection),"HostWorkSet::HostWorkSet::TRIPLE QUEUE MEM ALLOCATION");

	externSet<<<int(nVariableCollection*nQueen*nQueen)/1000+1,1000>>>(variablesMem,lastValuesMem,nQueen,nVariableCollection);
	ErrorChecking::hostErrorCheck(cudaPeekAtLastError(),"HostWorkSet::HostWorkSet::EXTERN SET CALL");
	ErrorChecking::hostErrorCheck(cudaDeviceSynchronize(),"HostWorkSet::HostWorkSet::SYNCH");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ HostWorkSet::~HostWorkSet(){
	ErrorChecking::hostErrorCheck(cudaFree(variablesMem),"HostWorkSet::~HostWorkSet::VARIABLES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(lastValuesMem),"HostWorkSet::~HostWorkSet::LAST VALUES MEM DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(tripleQueueMem),"HostWorkSet::~HostWorkSet::TRIPLE QUEUE ME DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariableCollection),"HostWorkSet::~HostWorkSet::DEVICE VARIABLE COLLECTION DEALLOCATION");
	ErrorChecking::hostErrorCheck(cudaFree(deviceVariable),"HostWorkSet::~HostWorkSet::DEVICE VARIABLE DEALLOCATION");
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
	DeviceQueenConstraints deviceQueenConstraints;
	DeviceQueenPropagation deviceQueenPropagation;

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

	__device__ int expand(int,int,int&);	//prepare for parallel computation on a specific level
											//for a chosen variable collection, return number of expansions
											//-1 otherwise

	__device__ int solve(int,int); 	//solve csp for all variable over a specific level
									//and returns the number of solutions.

	__device__ int solveAndAdd(int,int,int,DeviceParallelQueue&);
									//put in queue everything before level discriminant chosen
									//then it solve the csp 

	__device__ void print();

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceWorkSet::DeviceWorkSet(){}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void externInit(DeviceVariableCollection* deviceVariableCollection,
						   DeviceVariable* deviceVariable, int* variablesMem,
						   int* lastValuesMem, Triple* tripleQueueMem, int nQueen,
						   int nVariableCollection){

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
		ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceWorkSet::DeviceWorkSet::EXTERN INIT");

	}else{

		for(int i = 0; i < nVariableCollection*nQueen; ++i){

			deviceVariable[i].init2(&variablesMem[i*nQueen],nQueen);

			if(i < nVariableCollection){

				deviceVariableCollection[i].init2(&deviceVariable[i*nQueen],
												 &tripleQueueMem[i*nQueen*nQueen*3],
												 &variablesMem[i*nQueen*nQueen],
												 &lastValuesMem[i*nQueen],nQueen);

			}
		}
	}
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceWorkSet::DeviceWorkSet::SYNCH");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
		ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceWorkSet::init::EXTERN INIT");

	}else{

		for(int i = 0; i < nVariableCollection*nQueen; ++i){

			deviceVariable[i].init2(&variablesMem[i*nQueen],nQueen);

			if(i < nVariableCollection){

				deviceVariableCollection[i].init2(&deviceVariable[i*nQueen],
												 &tripleQueueMem[i*nQueen*nQueen*3],
												 &variablesMem[i*nQueen*nQueen],
												 &lastValuesMem[i*nQueen],nQueen);

			}
		}
	}
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceWorkSet::init::SYNCH");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceWorkSet::~DeviceWorkSet(){}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void externExpand(DeviceWorkSet& deviceWorkSet, int who, int count, int level, int nValues, int nQueen){

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	DeviceQueenPropagation deviceQueenPropagation;

	if(index < nQueen*nQueen*3*(nValues-1)){
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*count+index].var = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].var;
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*count+index].val = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].val;
		deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*count+index].cs = 
			deviceWorkSet.tripleQueueMem[nQueen*nQueen*3*who+(index%(nQueen*nQueen*3))].cs;
	}

	if(index < nQueen*nQueen*(nValues-1)){
		deviceWorkSet.variablesMem[nQueen*nQueen*count+index] = 
			deviceWorkSet.variablesMem[nQueen*nQueen*who+(index%(nQueen*nQueen))];
	}

	if(index < nQueen * (nValues-1)){
		deviceWorkSet.lastValuesMem[nQueen*count+index] = 
			deviceWorkSet.lastValuesMem[nQueen*who+(index%nQueen)];
	}

	if(index < (nValues-1)){
		deviceWorkSet.deviceVariableCollection[index+count].deviceQueue.count =
			deviceWorkSet.deviceVariableCollection[who].deviceQueue.count;
	}

	__syncthreads();

	int j = 0;
	int i = 0;
	if(index == 1){
		for(i = 0; i < nQueen && j<(nValues-1); ++i){
			if(deviceWorkSet.deviceVariableCollection[count+j].deviceVariable[level].domain[i] == 1){
				cudaStream_t s;
				ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"externExpand::STREAM CREATION");
				externAssignParallel<<<1,nQueen,0,s>>>(deviceWorkSet.deviceVariableCollection[count+j].deviceVariable[level].domain,nQueen,i);
				ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"externExpand::EXTERN ASSIGN PARALLEL CALL");
				deviceWorkSet.deviceVariableCollection[count+j].deviceVariable[level].ground = i;
				deviceWorkSet.deviceVariableCollection[count+j].lastValues[level] = i+1;
				ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"externExpand::STREAM DESTRUCTION");
				++j;
			}
		}
		for(; i < nQueen; ++i){
			if(deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].domain[i] == 1){

				cudaStream_t s;
				ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"externExpand::STREAM CREATION");
				externAssignParallel<<<1,nQueen,0,s>>>(deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].domain,nQueen,i);
				deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].ground = i;
				deviceWorkSet.deviceVariableCollection[who].lastValues[level] = i+1;

				break;
			}
		}
	}

	__syncthreads();

	if(index == nValues){
		deviceQueenPropagation.parallelForwardPropagation(
			deviceWorkSet.deviceVariableCollection[who],
			level,
			deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].ground);
	}
	if(index < nValues-1){
		deviceQueenPropagation.parallelForwardPropagation(
			deviceWorkSet.deviceVariableCollection[index+count],
			level,
			deviceWorkSet.deviceVariableCollection[index+count].deviceVariable[level].ground);
	}
}

__device__ int temp = 1;

__device__ int DeviceWorkSet::expand(int who, int level, int& oldCount){

	if(who < 0 || who >= count){
		ErrorChecking::deviceError("Error::DeviceWorkSet::expand::VARIABLE COLLECTION INDEX OUT OF BOUND");
		return -1;
	}

	if(level < 0 || level > nQueen){
		ErrorChecking::deviceError("Error::DeviceWorkSet::expand::LEVEL OUT OF BOUND");
		return -1;
	}

	int nValues = 0;
	for(int value = 0; value < nQueen; ++value)
		if(deviceVariableCollection[who].deviceVariable[level].domain[value] == 1)
			++nValues;

	while(atomicCAS(&lockCount,0,1)==1){}
	if((nValues-1) + count > nVariableCollection){
		ErrorChecking::deviceMessage("Warn::DeviceWorkSet::expand::NOT ENOUGH SPACE");
		lockCount = 0;
		return -1;
	}

	if(nValues == 0){
		ErrorChecking::deviceMessage("Warn::DeviceWorkSet::expand::VARIABLE IS FAILED");
		return 0;
	}

	atomicAdd(&temp,nValues-1);

	oldCount =  count;
	count += nValues-1;

	lockCount = 0;	

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceWorkSet::expand::STREAM CREATION");
	externExpand<<<int(nQueen*nQueen*3*nValues)/1000+1,1000,0,s>>>(*this,who,oldCount,level,nValues,nQueen);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceWorkSet::expand::EXTERN EXPAND CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceWorkSet::expand::STREAM DESTRUCTION");

	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceWorkSet::expand::SYNCH");

	return nValues-1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void DeviceWorkSet::print(){

	for(int i = 0; i < count; ++i){
		printf("------[%d]------\n", i);
		deviceVariableCollection[i].print();
	}
	printf("count : %d\n", count);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceWorkSet::solve(int who, int level){

	int ltemp = level - 1;
	int levelUp = 1;
	int val = 0;
	int nSols = 0;
	bool done = false;

	do{
		if(level == nQueen || deviceVariableCollection[who].isGround()){
			if(deviceQueenConstraints.solution(deviceVariableCollection[who],true)){
				++nSols;
			}
			deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
			--level;			
		}else{
			if(deviceVariableCollection[who].deviceVariable[level].ground < 0){
				val = deviceQueenPropagation.nextAssign(deviceVariableCollection[who],level);
				if(val == -1){
					if(level == 0){
						done = true;
					}else{
						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
						level -= levelUp;
						levelUp = 1;
					}
				}else{
					if(deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection[who],level,val)){
						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
						--level;
					}
					++level;
				}
			}else{
				++level;
				++levelUp;
			}
		}
		if(level == ltemp)done = true;
	}while(!done);
	return nSols;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceWorkSet::solveAndAdd(int who ,int level ,int levelDiscriminant, DeviceParallelQueue& deviceParallelQueue){

	int nSols = 0;

	return solve(who,level);

	int first = -1;

	while(level < levelDiscriminant){

		first = -1;

		for(int i = 0; i < nQueen; ++i){

			if(deviceVariableCollection[who].deviceVariable[level].domain[i] == 1){
				if(first == -1){
					first = i;
				}else{

					deviceVariableCollection[who].deviceVariable[level].assign(i);
					deviceVariableCollection[who].lastValues[level]=i+1;
					if(deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection[who],level,i)){

					}else if(deviceVariableCollection[who].isGround()){
						if(deviceQueenConstraints.solution(deviceVariableCollection[who],true)){
							++nSols;
						}
					}else{

						if(deviceParallelQueue.add(deviceVariableCollection[who],level+1,who)==-1){
							nSols += solve(who,level);
							deviceVariableCollection[who].deviceVariable[level].assign(i);
							deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection[who],level,i);

						}

					}
					deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
				}
			}
		}
		deviceVariableCollection[who].deviceVariable[level].assign(first);
		deviceVariableCollection[who].lastValues[level]=first+1;
		if(deviceQueenPropagation.parallelForwardPropagation(deviceVariableCollection[who],level,first)){
			break;
		}else if(deviceVariableCollection[who].isGround()){
			if(deviceQueenConstraints.solution(deviceVariableCollection[who],true)){
				++nSols;
				break;
			}
		}else if(level + 1 >= levelDiscriminant){
			nSols += solve(who,level+1);
			break;
		}
		++level;
	}



	return nSols;
}
