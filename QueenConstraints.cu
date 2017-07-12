#pragma once
#include "Variable.cu"

/*
	constraints to be checked only if VC is all ground
*/
struct QueenConstraints{

	int nVars;
	int nVals;

	__device__ QueenConstraints();			// doNothing
	__device__ QueenConstraints(int,int);	// just initialize nVars and nVals

	__device__ void init(int, int);			// just initialize nVars and nVals

	__device__ bool checkRowConstraint(Variable* vars);	//
	__device__ bool checkColConstraint(Variable* vars);	//specific implementation
	__device__ bool checkRDiagConstraint(Variable* vars);	//for queen problem
	__device__ bool checkLDiagConstraint(Variable* vars);	//

};


///////////////////////////////////////////////////////////////////////////////////////

__device__ QueenConstraints::QueenConstraints(){}

///////////////////////////////////////////////////////////////////////////////////////

__device__ QueenConstraints::QueenConstraints(int nvr, int nvl):nVars(nvr),nVals(nvl){}

///////////////////////////////////////////////////////////////////////////////////////

__device__ void QueenConstraints::init(int nvr, int nvl){
	nVars = nvr;
	nVals = nvl;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool QueenConstraints::checkRowConstraint(Variable* vars){
	int sum = 0;
	for(int j = 0; j < nVars; ++j){
		sum = 0;
		for(int i = 0; i < nVals; ++i){
			if(vars[j].domain[i] == 1)++sum;
		}
		if(sum != 1) return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool QueenConstraints::checkColConstraint(Variable* vars){

	int sum = 0;
	for(int j = 0; j < nVars; ++j){
		sum = 0;
		for(int i = 0; i <nVals; ++i){
			if(vars[i].domain[j] > 0)++sum;
		}
		if(sum != 1) return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool QueenConstraints::checkRDiagConstraint(Variable* vars){
	int sum,i,j,temp;

	for(j = 0; j < nVals; ++j){
		i = 0;
		sum = 0;
		temp=j;
		while(j < nVals && i < nVars){
			if(vars[i].domain[j]==1)++sum;
			++j;
			++i;
		}
		j = temp;
		if(sum < 0 || sum > 1) return false;
	}

	for(i = 1; i < nVars; ++i){
		j = 0;
		sum = 0;
		temp = i;
		while(j < nVals && i < nVars){
			if(vars[i].domain[j]==1)++sum;
			++j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool QueenConstraints::checkLDiagConstraint(Variable* vars){
	int sum,i,j,temp;

	for(j = 0; j < nVals; ++j){
		i = 0;
		sum = 0;
		temp = j;
		while(j >= 0 && i < nVars){
			if(vars[i].domain[j]==1)++sum;
			--j;
			++i;
		}
		j = temp;
		if(sum < 0 || sum > 1) return false;
	}

	for(i = 1; i < nVars; ++i){
		j = nVals-1;
		sum = 0;
		temp = i;
		while(j >= 0 && i < nVars){
			if(vars[i].domain[j]==1)++sum;
			--j;
			++i;
		}
		i = temp;
		if(sum < 0 || sum > 1) return false;
	}

	return true;
}




















