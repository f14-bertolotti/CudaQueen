#include <stdio.h>
#include "Variable.cu"
#include "VariableCollection.cu"
#include "QueenConstraints.cu"
#include "TripleQueue.cu"
#include "QueenPropagation.cu"
#include "WorkSet.cu"

__device__ const int nVars = 7;
__device__ const int nVals = 7;

__global__ void test(){

	int* lv = (int*)malloc(sizeof(int)*nVars);
	int* ptr = (int*)malloc(sizeof(int)*nVars*nVals);
	Triple* ptrtr = (Triple*)malloc(sizeof(Triple)*3*nVars*nVars);

	Variable vs[nVars];
	VariableCollection vc;
	QueenConstraints qc(nVars,nVals);
	TripleQueue tq;
	QueenPropagation qp;
	WorkSet ws;


	for(int i = 0; i < nVars; ++i){
		vs[i].init(&ptr[nVars*i],nVals);
	}
	vc.init(vs,&qc,nVars,nVals);
	tq.init(ptrtr,nVars,nVals);
	qp.init(&vc,lv,&tq,nVars,nVals);
	ws.init(&vc,&qc,&qp,nVars,nVals);
	
	printf("nSols: %d\n",ws.solveAll(0));
	

	free(ptr);
	free(lv);
	free(ptrtr);
}


int main(){

	test<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;

}
