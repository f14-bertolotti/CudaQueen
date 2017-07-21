#pragma once

#include <stdio.h>
#include "../Variable/Variable.cu"
#include "../QueenConstraints/QueenConstraints.cu"


struct VariableCollection{

	bool dbg;		//if on is verbose
	int nVars;
	int nVals;
	Variable* vars;
	QueenConstraints* qc;

	__device__ VariableCollection();
	__device__ VariableCollection(Variable*,QueenConstraints*,int,int);	//inits

	__device__ void init(Variable*,QueenConstraints*,int,int);			//inits

	__device__ void assign(int,int);	//assign for a specific var and val

	__device__ bool isGround();			//check if v.c. is ground
	__device__ bool isSolution();		//check if v.c. is in solution
	__device__ bool isFailed();			//check if v.c. is in failed state

	__device__ void print(int); // print with modes

	__device__ ~VariableCollection();

};

///////////////////////////////////////////////////////////////////////////////////////

__device__ VariableCollection::VariableCollection(){}

///////////////////////////////////////////////////////////////////////////////////////

__device__ VariableCollection::VariableCollection(Variable* doms, QueenConstraints* qcptr,
		int nvr, int nvl):vars(doms), qc(qcptr), nVars(nvr),nVals(nvl){
	dbg = false;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ void VariableCollection::init(Variable* doms, QueenConstraints* qcptr, int nvr, int nvl){
	nVars = nvr;
	nVals = nvl;
	vars = doms;
	qc = qcptr;

	dbg = false;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ void VariableCollection::assign(int var, int val){
	vars[var].assign(val);
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool VariableCollection::isGround(){
	for(int i = 0; i < nVars; ++i)
		if(vars[i].ground==-1)return false;

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool VariableCollection::isSolution(){
	if(!qc->checkRowConstraint(vars))return false;
	if(!qc->checkColConstraint(vars))return false;
	if(!qc->checkRDiagConstraint(vars))return false;
	if(!qc->checkLDiagConstraint(vars))return false;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ bool VariableCollection::isFailed(){
	for(int i = 0; i < nVars; ++i)
		if(vars[i].failed == 1)return true;

	return false;
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ void VariableCollection::print(int mode){
	for(int i = 0; i < nVars; ++i)
		vars[i].print(mode);
}

///////////////////////////////////////////////////////////////////////////////////////

__device__ VariableCollection::~VariableCollection(){}

///////////////////////////////////////////////////////////////////////////////////////














