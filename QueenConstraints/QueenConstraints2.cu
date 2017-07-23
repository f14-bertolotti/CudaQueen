#pragma once
#include "../VariableCollection/VariableCollection2.cu"

/*
	constraints to be checked only if VC is all ground
*/
struct DeviceQueenConstraints{

	//////////////////////////////////////SINGLE THREAD//////////////////////////////////////

	__device__ bool checkRowConstraint(DeviceVariableCollection&);		//
	__device__ bool checkColConstraint(DeviceVariableCollection&);		//specific implementation
	__device__ bool checkRDiagConstraint(DeviceVariableCollection&);	//for queen problem
	__device__ bool checkLDiagConstraint(DeviceVariableCollection&);	//

	//////////////////////////////////////MULTI THREAD//////////////////////////////////////

	__device__ bool parallelConstraints(DeviceVariableCollection&);		//specific for queen problem
};

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueenConstraints::checkRowConstraint(DeviceVariableCollection& vc){
	int sum = 0;
	for(int j = 0; j < vc.nQueen; ++j){
		sum = 0;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.variables[j].domain[i] == 1)++sum;
		}
		if(sum != 1) return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueenConstraints::checkColConstraint(DeviceVariableCollection& vc){

	int sum = 0;
	for(int j = 0; j < vc.nQueen; ++j){
		sum = 0;
		for(int i = 0; i <vc.nQueen; ++i){
			if(vc.variables[i].domain[j] > 0)++sum;
		}
		if(sum != 1) return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueenConstraints::checkRDiagConstraint(DeviceVariableCollection& vc){
	int sum,i,j,temp;

	for(j = 0; j < vc.nQueen; ++j){
		i = 0;
		sum = 0;
		temp=j;
		while(j < vc.nQueen && i < vc.nQueen){
			if(vc.variables[i].domain[j]==1)++sum;
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
			if(vc.variables[i].domain[j]==1)++sum;
			++j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueenConstraints::checkLDiagConstraint(DeviceVariableCollection& vc){
	int sum,i,j,temp;

	for(j = 0; j < vc.nQueen; ++j){
		i = 0;
		sum = 0;
		temp = j;
		while(j >= 0 && i < vc.nQueen){
			if(vc.variables[i].domain[j]==1)++sum;
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
			if(vc.variables[i].domain[j]==1)++sum;
			--j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool okDiags = true;
__global__ void externParallelDiagConstr(int* Mem, int nQueen){
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
			if(sum > 1)
				okDiags = false;					
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
	
			if(sum > 1)
				okDiags = false;
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

			if(sum > 1)
				okDiags = false;
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
			if(sum > 1)
				okDiags = false;
			break;
		}
	}

	__syncthreads();

}

__device__ bool okAllDiffs = true;
__global__ void externParallelAllDiffs(int* Mem, int nQueen){
	int sum = 0;
	for(int i = 0 ; i < nQueen; ++i)
		if(Mem[i*nQueen+threadIdx.x]==1)
			++sum;
	
	if(sum != 1)
		okAllDiffs = false;

	__syncthreads();

}

///////////////////////////////////////////////////////////////////////

__device__ bool DeviceQueenConstraints::parallelConstraints(DeviceVariableCollection& vc){
	externParallelAllDiffs<<<1,vc.nQueen>>>(vc.deviceMemoryManagement.dMem,vc.nQueen);
	externParallelDiagConstr<<<1,vc.nQueen*4>>>(vc.deviceMemoryManagement.dMem,vc.nQueen);
	cudaDeviceSynchronize();
	bool res = okAllDiffs && okDiags;
	okAllDiffs = true;
	okDiags = true;
	return res;
}















