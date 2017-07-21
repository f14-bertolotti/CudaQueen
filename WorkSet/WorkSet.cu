#pragma once
#include <stdio.h>
#include "../QueenConstraints/QueenConstraints.cu"
#include "../QueenPropagation/QueenPropagation.cu"
#include "../VariableCollection/VariableCollection.cu"

struct WorkSet{
	int nVars;
	int nVals;

	VariableCollection* vc;	//contains variables
	QueenConstraints* qc;	//contains constraint
	QueenPropagation* qp;	//apply assignement and propagation

	__device__ WorkSet();
	__device__ WorkSet(VariableCollection*,QueenConstraints*,QueenPropagation*,int,int);

	__device__ void init(VariableCollection*,QueenConstraints*,QueenPropagation*,int,int);

	__device__ int numberOfAssign(int);		//number to assignemet (upper bound) to complete search
											//starts from level
	__device__ int solveAll(int);			//solve csp return number of solutions
											//starts from level
	__device__ void copy(WorkSet& ws);		//just makes a copy (from father to child)
	__device__ void print();				//prints states

};

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ WorkSet::WorkSet(){}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int WorkSet::numberOfAssign(int level){
	int sum = 0;
	for(int i = level; i < nVars; ++i)
		for(int j = 0; j < nVals; ++j)
			if(vc->vars[i].domain[j]==1)
				++sum;

	return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void WorkSet::init(VariableCollection* vct,QueenConstraints* qct,QueenPropagation* qpt,
		int nvr, int nvl){
	
	nVars = nvr;
	nVals = nvl;
	qp = qpt;  	
	qc = qct;
	vc = vct;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int WorkSet::solveAll(int level){


	int ltemp = level-1;
	int val = 0;
	int nSols = 0;
	bool done = false;
	do{
		if(level == nVars){
			if(vc->isSolution()){
				++nSols;
			}
			qp->undoForwardPropagation();
			--level;	
		
		}else{

			val = qp->nextAssign(level);
			if(val == -1){
				if(level == 0) done = true;
				else{
					qp->undoForwardPropagation();
					--level;
				}
			}else{
				qp->forwardPropagation(level,val);
				if(vc->isFailed()){
					qp->undoForwardPropagation();
					--level;
				}
				++level;
			}
		}

	if(level == ltemp) done = true;

	}while(!done);

	return nSols;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void WorkSet::print(){

	vc->print(2);
	qp->printVisited();
	qp->queue->print();

}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void WorkSet::copy(WorkSet& ws){
	for(int i = 0; i < nVars; ++i){
		qp->lastValues[i]=ws.qp->lastValues[i];
		for(int j = 0; j < nVals; ++j){
			vc->vars[i].domain[j]=ws.vc->vars[i].domain[j];
		}
	}
	for(int i = 0; i < ws.qp->queue->count; ++i)
		qp->queue->add(ws.qp->queue->q[i].var,ws.qp->queue->q[i].val,ws.qp->queue->q[i].cs);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

