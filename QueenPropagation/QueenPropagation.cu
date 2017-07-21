#pragma once
#include <stdio.h>
#include "../Variable/Variable.cu"
#include "../VariableCollection/VariableCollection.cu"
#include "../TripleQueue/TripleQueue.cu"

struct QueenPropagation{

	int nVars;
	int nVals;
	TripleQueue* queue;			//queue of propagation applied
	VariableCollection* vc;		//V.C.
	int* lastValues;			//maintains track of value already assigned

	bool dbg;		//if on is verbose

	__device__ QueenPropagation();														//do nothing
	__device__ QueenPropagation(VariableCollection* vars,int*,TripleQueue*,int,int);	//initialize queue

	__device__ void init(VariableCollection* vars,int*,TripleQueue*,int,int);			//initialize queue

	__device__ int nextAssign(int);		//assign next value not already tried
										//returns assigned value

	__device__ void allDifferent(int,int,int);	//propagate for all different constraint code 3
	__device__ void diagDifferent(int,int,int);	//propagate for diag constraint code 4

	__device__ void forwardPropagation(int,int);	//csp forward propagation code 5
	__device__ void undoForwardPropagation();		//csp undo forward propagation code 5

	__device__ void printVisited();				//print lastValues

	__device__ ~QueenPropagation();				//do nothing

};

////////////////////////////////////////////////////////////////////////////

__device__ QueenPropagation::QueenPropagation(){}

////////////////////////////////////////////////////////////////////////////

__device__ QueenPropagation::QueenPropagation(VariableCollection* vars, int* lv,
		TripleQueue* tq, int nvr, int nvl):
		vc(vars),lastValues(lv),nVars(nvr),nVals(nvl),queue(tq){

	for(int i = 0; i < nVars; ++i)
		lastValues[i]=0;
	dbg = false;
}

////////////////////////////////////////////////////////////////////////////

__device__ void QueenPropagation::init(VariableCollection* vars, int* lv,
		TripleQueue* tq, int nvr, int nvl){

	vc = vars;
	lastValues = lv;
	nVars = nvr;
	nVals = nvl;
	queue = tq;

	for(int i = 0; i < nVars; ++i)lastValues[i]=0;
	dbg = false;
}


////////////////////////////////////////////////////////////////////////////

__device__ int QueenPropagation::nextAssign(int var){
	Variable* vars = vc->vars;

	if(var<0 || var>=nVars){
		printf("Error::QueenPropagation::nextAssign::VAR OUT OF BOUND\n");
		return -1;
	}

	if(lastValues[var] >= nVals){
		if(dbg)printf("Msg::QueenPropagation::nextAssign::VALUE OUT OF BOUND\n");
		return -1;
	}

	if(vars[var].failed == 1){
		printf("Error::QueenPropagation::nextAssign::VAR ALREADY FAILED\n");
		return -1;
	}

	int next;
	for(next = lastValues[var];next<nVars;++next)
		if(vars[var].domain[next]==1){
			lastValues[var]=next+1;
			vc->assign(var,next);
			return next;
		}

	if(dbg)printf("Msg::QueenPropagation::nextAssign::NEXTVALUE NOT FOUND\n");
	return -1;
}

////////////////////////////////////////////////////////////////////////////

__device__ void QueenPropagation::allDifferent(int var, int val, int delta){
	if(var < 0 || var > nVars || val < 0 || val > nVals){
		printf("Error::QueenPropagation::allDifferent::OUT OF BOUND\n");
		return;
	}

	Variable* vars=vc->vars;

	if(vars[var].ground != val){
		printf("Error::QueenPropagation::allDifferent::VARIABLE NOT GROUND\n");
		return;
	}
	
	for(int i = 0; i < nVars; ++i)
		if(i != var){
			vars[i].addTo(val,delta);

		}
	
	if(delta < 0)queue->add(var,val,3);
	
}

////////////////////////////////////////////////////////////////////////////


__device__ void QueenPropagation::diagDifferent(int var, int val, int delta){
	if(var < 0 || var > nVars || val < 0 || val > nVals){
		printf("Error::QueenPropagation::diagDifferent::OUT OF BOUND\n");
		return;
	}

	Variable* vars=vc->vars;

	if(vars[var].ground != val){
		printf("Error::QueenPropagation::diagDifferent::VARIABLE NOT GROUND\n");
		return;
	}

	int i=var+1,j=val+1;
	while(i<nVars && j<nVals){
		vars[i].addTo(j,delta);
		++i;++j;
	}

	i=var-1,j=val-1;
	while(i>=0 && j>=0){
		vars[i].addTo(j,delta);
		--i;--j;
	}

	i=var-1,j=val+1;
	while(i>=0 && j<nVals){
		vars[i].addTo(j,delta);
		--i;++j;
	}

	i=var+1,j=val-1;
	while(i<nVars && j>=0){
		vars[i].addTo(j,delta);
		++i;--j;
	}

	if(delta < 0)queue->add(var,val,4);
}

////////////////////////////////////////////////////////////////////////////

__device__ void QueenPropagation::forwardPropagation(int var, int val){

	if(var < 0 || var > nVars){
		printf("Error::QueenPropagation::forwardPropagation:: VAR OUT OF BOUND\n");
		return;
	}

	if(val < 0 || val > nVals){
		printf("Error::QueenPropagation::forwardPropagation:: VAL OUT OF BOUND\n");
		return;
	}

	Variable* vars=vc->vars;

	if(vars[var].ground != val){
		printf("Error::QueenPropagation::forwardPropagation::VARIABLE NOT GROUND\n");
		return;
	}

	allDifferent(var,val,-1);
	diagDifferent(var,val,-1);

	/*printf("---in propagation---\n");
	vc->print(2);
	printf("--------------------\n");*/

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < nVars; ++i){
			if(vars[i].changed==1){
				if(vars[i].ground>=0){
					allDifferent(i,vars[i].ground,-1);
					diagDifferent(i,vars[i].ground,-1);
					ch = true;
				}
				vars[i].changed=-1;
			}
		}
		
		/*printf("---in propagation---\n");
		vc->print(2);
		printf("--------------------\n");*/

	}while(ch);

	queue->add(var,val,5);

}

////////////////////////////////////////////////////////////////////////////

__device__ void QueenPropagation::undoForwardPropagation(){
	if(queue->front()->cs!=5){
		printf("Error::QueenPropagation::undoForwardPropagation::ERROR IN QUEUE\n");
		return;		
	}

	if(queue->empty()){
		printf("Error::QueenPropagation::undoForwardPropagation::EMPTY QUEUE\n");
		return;		
	}

	int t1=queue->front()->var;
	int t2=queue->front()->val;

	for(int i = t1+1; i < nVars; ++i)lastValues[i]=0;

	queue->pop();
	while(queue->front()->cs!=5){
		switch(queue->front()->cs){
			case 3:{
				allDifferent(queue->front()->var,queue->front()->val,+1);	
			}break;
			case 4:{
				diagDifferent(queue->front()->var,queue->front()->val,+1);	
			}break;
		}
		queue->pop();

		/*printf("---in propagation---\n");
		vc->print(2);
		printf("--------------------\n");
*/
		if(queue->empty())break;
	}

	vc->vars[t1].undoAssign(t2);
}

////////////////////////////////////////////////////////////////////////////

__device__ void QueenPropagation::printVisited(){
	printf("visited:: ");
	for(int i = 0; i < nVars; ++i){
		printf("%d ",lastValues[i]);
	}printf("\n");
}

////////////////////////////////////////////////////////////////////////////

__device__ QueenPropagation::~QueenPropagation(){}

////////////////////////////////////////////////////////////////////////////









