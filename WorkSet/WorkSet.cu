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

	__device__ int expand(int,int&,int&);	//prepare for parallel computation on a specific level
											//for a chosen variable collection, return number of expansions
											//-1 otherwise

	__device__ int solve(int,int,int&,int,int*); 	//solve csp for all variable over a specific level
									//and returns the number of solutions.

	__device__ int solveAndAdd(int,int,int&,int,int*,int,DeviceParallelQueue&);
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

__device__ void externExpand(DeviceWorkSet& deviceWorkSet, int who, int count, int& level, int& nValues, int nQueen){

	__shared__ int sharedCount;
	sharedCount = count;


	__syncthreads();
//	if(threadIdx.x == 0)printf("%d - %d\n", sharedCount,nValues);

	DeviceQueenPropagation deviceQueenPropagation;

	__syncthreads();

	for (int i = sharedCount; i < sharedCount+nValues-1; ++i){
__syncthreads();
		deviceWorkSet.deviceVariableCollection[i] = deviceWorkSet.deviceVariableCollection[who];
__syncthreads();
	}
	
	__syncthreads();

	int j = 0;
	int i = 0;


	__syncthreads();


	for(; i < nQueen && j < nValues-1;++i){

		__syncthreads();

		if(deviceWorkSet.deviceVariableCollection[sharedCount+j].deviceVariable[level].domain[i] == 1){
			
			__syncthreads();

			if(threadIdx.x < nQueen && threadIdx.x != i)
				--deviceWorkSet.deviceVariableCollection[sharedCount+j].deviceVariable[level].domain[threadIdx.x];
			deviceWorkSet.deviceVariableCollection[sharedCount+j].deviceVariable[level].ground = i;
			deviceWorkSet.deviceVariableCollection[sharedCount+j].lastValues[level] = i+1;

			++j;

		}

		__syncthreads();
	}

__syncthreads();

	for(;i < nQueen;++i){

		__syncthreads();

		if(deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].domain[i] == 1){
			__syncthreads();
			if(threadIdx.x < nQueen && threadIdx.x != i)
				--deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].domain[threadIdx.x];
			deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].ground = i;
			deviceWorkSet.deviceVariableCollection[who].lastValues[level] = i+1;
__syncthreads();
			break;
		}

		__syncthreads();

	}
__syncthreads();
	
	for(int i = sharedCount; i < sharedCount+nValues-1; ++i){
__syncthreads();
			deviceQueenPropagation.parallelForwardPropagation2(
				deviceWorkSet.deviceVariableCollection[i],
				level,
				deviceWorkSet.deviceVariableCollection[i].deviceVariable[level].ground);
			__syncthreads();
	}
__syncthreads();
	deviceQueenPropagation.parallelForwardPropagation2(
		deviceWorkSet.deviceVariableCollection[who],
		level,
		deviceWorkSet.deviceVariableCollection[who].deviceVariable[level].ground);
__syncthreads();
	return;

}


__device__ int DeviceWorkSet::expand(int who, int& level, int& oldCount){

	if(who < 0 || who >= count){
		ErrorChecking::deviceError("Error::DeviceWorkSet::expand::VARIABLE COLLECTION INDEX OUT OF BOUND");
		return -1;
	}

	if(level < 0 || level > nQueen){
		ErrorChecking::deviceError("Error::DeviceWorkSet::expand::LEVEL OUT OF BOUND");
		return -1;
	}
__syncthreads();
	__shared__ int nValues;
__syncthreads();
	nValues = 0;

	__syncthreads();

	if(threadIdx.x == 0){
		for(int i = 0; i < nQueen; ++i){
			if(deviceVariableCollection[who].deviceVariable[level].domain[i] == 1){
				++nValues;
			}
		}
	}

	/*if(threadIdx.x < nQueen && deviceVariableCollection[who].deviceVariable[level].domain[threadIdx.x] == 1){
		atomicAdd(&nValues,1);
	}*/

	__syncthreads();

	if(nValues <= 1)return 0;


	__shared__ int t;

	__syncthreads();

	if(threadIdx.x == 0){
		t = atomicAdd(&count, nValues-1);
	}

	__syncthreads(); 
	
	oldCount = t;

	__syncthreads();

	if((nValues-1) + count > nVariableCollection){
		ErrorChecking::deviceMessage("Warn::DeviceWorkSet::expand::NOT ENOUGH SPACE");
		return -1;
	}

__syncthreads();

	if(nValues == 0){
		ErrorChecking::deviceMessage("Warn::DeviceWorkSet::expand::VARIABLE IS FAILED");
		return 0;
	}

__syncthreads();



__syncthreads();

	externExpand(*this,who,t,level,nValues,nQueen);

__syncthreads();

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

__device__ int DeviceWorkSet::solve(int who, int outLevel, int& nodes, int count, int* nodesPerBlock){

	__shared__ int level;
	__shared__ int ltemp;
	__shared__ int levelUp;
	__shared__ int val;
	__shared__ int nSols;
	__shared__ bool done;

	__syncthreads();


	level = outLevel;
	ltemp = outLevel-1;
	levelUp = 1;
	val = 0;
	nSols = 0;
	done = false;
__syncthreads();
	do{

		__syncthreads();
		if(deviceVariableCollection[who].isGround()){
			if(threadIdx.x == 0)++nSols;
			__syncthreads();
			deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
			if(threadIdx.x == 0){
				--level;
			}	

		}else{
			__syncthreads();

			if(deviceVariableCollection[who].deviceVariable[level].ground < 0){
				__syncthreads();

				val = deviceQueenPropagation.nextAssign(deviceVariableCollection[who],level);
				
__syncthreads();

				if(val == -1){
__syncthreads();

					if(level <= ltemp+1){
__syncthreads();		

						done = true;
__syncthreads();
					}else{
__syncthreads();

						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
__syncthreads();

						if(threadIdx.x == 0){
							level -= levelUp;
							levelUp = 1;

						}
__syncthreads();

					}
__syncthreads();

				}else{

__syncthreads();
					if(deviceQueenPropagation.parallelForwardPropagation2(deviceVariableCollection[who],level,val)){
						__syncthreads();
						deviceQueenPropagation.parallelUndoForwardPropagation(deviceVariableCollection[who]);
__syncthreads();

						if(threadIdx.x == 0){
							--level;
						}
__syncthreads();

					}
					__syncthreads();

					if(threadIdx.x == 0)++level;
__syncthreads();

				}

__syncthreads();
			}else{
				__syncthreads();

				if(threadIdx.x == 0){
					++level;
					++levelUp;
				}
__syncthreads();
			}
__syncthreads();
		}
__syncthreads();
		__syncthreads();

		if(level == ltemp){
			done = true;
		}

		__syncthreads();
	}while(!done);

__syncthreads();
	return nSols;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ int DeviceWorkSet::solveAndAdd(int who ,int level , int& nodes, int count, int* nodesPerBlock, int levelDiscriminant, DeviceParallelQueue& deviceParallelQueue){

	int nSols = 0;


	if(deviceVariableCollection[who].isFailed()){
		return 0;
	}else if(deviceVariableCollection[who].isGround()){
		if(deviceQueenConstraints.solution(deviceVariableCollection[who])){
			return 1;
		}
		return 0;
	}

	while(level < levelDiscriminant){
		//fino a che il livello è minore del secondo dicriminante

		while(deviceVariableCollection[who].deviceVariable[level].ground >= 0){
			++level;
			if(level >= levelDiscriminant){
				return nSols + solve(who,level,nodes,count,nodesPerBlock);
			}
		}


		int nVal = deviceParallelQueue.expansion(deviceVariableCollection[who],level);
		if(nVal != -1){
			atomicAdd(&nodes, nVal);
			nodesPerBlock[count] += nVal;
		}else return nSols + solve(who,level,nodes,count,nodesPerBlock);

		if(deviceVariableCollection[who].isFailed()){
			//il primo è fallito
			return nSols;
		}else if(deviceVariableCollection[who].isGround()){
			if(deviceQueenConstraints.solution(deviceVariableCollection[who])){
				return nSols + 1;
			}
			return nSols;
		}

		++level;

		if(level >= levelDiscriminant)
			return nSols + solve(who,level,nodes,count,nodesPerBlock);
	}


	return nSols;
}
