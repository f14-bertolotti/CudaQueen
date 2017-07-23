#pragma once
#include "../VariableCollection/VariableCollection2.cu"

/*
	constraints to be checked only if VC is all ground
*/
struct DeviceQueenConstraints{

	__device__ bool checkRowConstraint(DeviceVariableCollection&);		//
	__device__ bool checkColConstraint(DeviceVariableCollection&);		//specific implementation
	__device__ bool checkRDiagConstraint(DeviceVariableCollection&);	//for queen problem
	__device__ bool checkLDiagConstraint(DeviceVariableCollection&);	//

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




















