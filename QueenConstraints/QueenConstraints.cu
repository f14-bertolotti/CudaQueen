#pragma once
#include "../VariableCollection/VariableCollection.cu"
#include "../ErrorChecking/ErrorChecking.cu"

/*
	constraints to be checked only if VC is all ground
*/
struct DeviceQueenConstraints{

	//////////////////////////////////////SINGLE THREAD//////////////////////////////////////

	__device__ bool static inline checkRowConstraint(DeviceVariableCollection&);		//
	__device__ bool static inline checkColConstraint(DeviceVariableCollection&);		//specific implementation
	__device__ bool static inline checkRDiagConstraint(DeviceVariableCollection&);	//for queen problem
	__device__ bool static inline checkLDiagConstraint(DeviceVariableCollection&);	//
																		
	//////////////////////////////////////MULTI THREAD//////////////////////////////////////

	__device__ bool static inline parallelConstraints(DeviceVariableCollection&);		//specific for queen problem

	////////////////////////////////////////////////////////////////////////////////////////

	__device__ bool static inline solution(DeviceVariableCollection&,bool);			//check solution	
};

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::checkRowConstraint(DeviceVariableCollection& vc){

	int sum = 0;
	for(int j = 0; j < vc.nQueen; ++j){
		sum = 0;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.deviceVariable[j].domain[i] == 1)++sum;
		}
		if(sum != 1) return false;
	}

	return true;

}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::checkColConstraint(DeviceVariableCollection& vc){

	int sum = 0;
	for(int j = 0; j < vc.nQueen; ++j){
		sum = 0;
		for(int i = 0; i <vc.nQueen; ++i){
			if(vc.deviceVariable[i].domain[j] > 0)++sum;
		}
		if(sum != 1) return false;
	}

	return true;

}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::checkRDiagConstraint(DeviceVariableCollection& vc){

	int sum,i,j,temp;

	for(j = 0; j < vc.nQueen; ++j){
		i = 0;
		sum = 0;
		temp=j;
		while(j < vc.nQueen && i < vc.nQueen){
			if(vc.deviceVariable[i].domain[j]==1)++sum;
			++j;
			++i;
		}
		j = temp;
		if(sum < 0 || sum > 1) return false;
	}

	for(i = 1; i < vc.nQueen; ++i){
		j = 0;
		sum = 0;
		temp = i;
		while(j < vc.nQueen && i < vc.nQueen){
			if(vc.deviceVariable[i].domain[j]==1)++sum;
			++j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}
	return true;

}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::checkLDiagConstraint(DeviceVariableCollection& vc){

	int sum,i,j,temp;

	for(j = 0; j < vc.nQueen; ++j){
		i = 0;
		sum = 0;
		temp = j;
		while(j >= 0 && i < vc.nQueen){
			if(vc.deviceVariable[i].domain[j]==1)++sum;
			--j;
			++i;
		}
		j = temp;
		if(sum < 0 || sum > 1) return false;
	}

	for(i = 1; i < vc.nQueen; ++i){
		j = vc.nQueen-1;
		sum = 0;
		temp = i;
		while(j >= 0 && i < vc.nQueen){
			if(vc.deviceVariable[i].domain[j]==1)++sum;
			--j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}

	return true;

}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::solution(DeviceVariableCollection& vc, bool fullParallel){

	if(fullParallel) return parallelConstraints(vc);
	else return checkRowConstraint(vc) && checkColConstraint(vc) && checkRDiagConstraint(vc) && checkLDiagConstraint(vc);

}

///////////////////////////////////////////////////////////////////////////////////////

__global__ void externParallelDiagConstr(int* Mem, int nQueen, bool* okDiags){

	int sum,i,j,what;

	if(threadIdx.x < nQueen)what = 0;
	else if(threadIdx.x >= nQueen && threadIdx.x<2*nQueen)what = 1;
	else if(threadIdx.x >= 2*nQueen && threadIdx.x<3*nQueen)what = 2;
	else if(threadIdx.x >= 3*nQueen && threadIdx.x<4*nQueen)what = 3;

	switch(what){
		case 0:{
			j = threadIdx.x % nQueen;
			i = 0;
			sum = 0;
			while(j < nQueen && i < nQueen){
				if(Mem[nQueen*i + j]==1)++sum;
				++j;
				++i;
			}
			if(sum > 1){
				*okDiags = false;					
			}
			break;
		}
		case 1:{

			i = threadIdx.x % nQueen;
			j = 0;
			sum = 0;
			while(j < nQueen && i < nQueen){
				if(Mem[nQueen*i + j]==1)++sum;
				++j;
				++i;
			}
	
			if(sum > 1){
				*okDiags = false;
			}
			break;
		}
		case 2:{

			j = threadIdx.x % nQueen;
			i = 0;
			sum = 0;
			while(j >= 0 && i < nQueen){
				if(Mem[nQueen*i + j]==1)++sum;
				--j;
				++i;
			}

			if(sum > 1){
				*okDiags = false;
			}
			break;
		}
		case 3:{
			i = threadIdx.x % nQueen;
			j = nQueen-1;
			sum = 0;
			while(j >= 0 && i < nQueen){
				if(Mem[nQueen*i + j]==1)++sum;
				--j;
				++i;
			}
			if(sum > 1){
				*okDiags = false;
			}
			break;
		}
	}

}

__global__ void externParallelAllDiffs(int* Mem, int nQueen, bool* okAllDiffs){

	int sum = 0;
	for(int i = 0 ; i < nQueen; ++i){
		if(Mem[i*nQueen+threadIdx.x]==1)
			++sum;
	}
	
	if(sum != 1){
		*okAllDiffs = false;
	}

}

///////////////////////////////////////////////////////////////////////

__device__ bool inline DeviceQueenConstraints::parallelConstraints(DeviceVariableCollection& vc){

	cudaStream_t s1,s2;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking),"DeviceQueenConstraints::parallelConstraints::STREAM CREATION 1");
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking),"DeviceQueenConstraints::parallelConstraints::STREAM CREATION 2");
	__shared__	bool res1, res2;
	res1 = res2 = true;
	externParallelAllDiffs<<<1,vc.nQueen,0,s1>>>(vc.dMem,vc.nQueen,&res1);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenConstraints::parallelConstraints::EXTERN PARALLEL ALL DIFFS");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s1),"DeviceQueenConstraints::parallelConstraints::STREAM DESTRUCTION 1");
	externParallelDiagConstr<<<1,vc.nQueen*4,0,s2>>>(vc.dMem,vc.nQueen,&res2);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenConstraints::parallelConstraints::EXTERN PARALLEL DIAGS DIFFS");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s2),"DeviceQueenConstraints::parallelConstraints::STREAM DESTRUCTION 2");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenConstraints::parallelConstraints::SYNCH");
	return res1 && res2;;

}

///////////////////////////////////////////////////////////////////////














