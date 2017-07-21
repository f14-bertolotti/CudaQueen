#include <stdio.h>
#include "./Variable/Variable.cu"
#include "./VariableCollection/VariableCollection.cu"
#include "./QueenConstraints/QueenConstraints.cu"
#include "./QueenPropagation/QueenPropagation.cu"
#include "./TripleQueue/TripleQueue.cu"
#include "./WorkSet/WorkSet.cu"

////////////////////////////////////////////////////////////////////////////////////////////

__device__ const int nVars=8;
__device__ const int nVals=8;

__device__ QueenConstraints qc;
__device__ Variable vs[nVars];
__device__ VariableCollection vc;
__device__ TripleQueue tq;
__device__ QueenPropagation qp;
__device__ WorkSet ws;

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

	int level = 0;
	int val = 0;
	int nSols = 0;
	bool done = false;
	do{
		if(level == nVars){
			if(vc.isSolution()){
				++nSols;
			}
			qp.undoForwardPropagation();
			--level;			
		}else{

			val = qp.nextAssign(level);
			if(val == -1){
				if(level == 0) done = true;
				else{
					qp.undoForwardPropagation();
					--level;
				}
			}else{
				qp.forwardPropagation(level,val);
				if(vc.isFailed()){
					qp.undoForwardPropagation();
					--level;
				}
				++level;
			}
		}
	}while(!done);

	printf("\033[32mSOLUTIONS FOUND = %d\033[0m\n",nSols);
}

////////////////////////////////////////////////////////////////////////////////////////////

void init();
__global__ void init(int*,int*,Triple*);

////////////////////////////////////////////////////////////////////////////////////////////

int main(){


	init();

    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
	float   elapsedTime;
	cudaEventRecord( start, 0 );

	test<<<1,1>>>();
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\033[36mTIME: %f\033[0m\n", elapsedTime);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(int* dLastValuesMem, int* dVarsMem, Triple* dQueueMem){

	qc.init(nVars,nVals);
	for(int i = 0; i < nVars; ++i)
		vs[i].init(&dVarsMem[nVars*i],nVals);
	vc.init(vs,&qc,nVars,nVals);
	tq.init(dQueueMem,nVars,nVals);
	qp.init(&vc,dLastValuesMem,&tq,nVars,nVals);
	ws.init(&vc,&qc,&qp,nVars,nVals);
	
}

////////////////////////////////////////////////////////////////////////////////////////////

void init(){

	/*device allocation*/
	int* dLastValuesMem = NULL;
	int* dVarsMem = NULL;
	Triple* dQueueMem = NULL;

	cudaMalloc((void**)&dLastValuesMem,sizeof(int)*nVars);
	cudaMalloc((void**)&dVarsMem,sizeof(int)*nVars*nVals);
	cudaMalloc((void**)&dQueueMem,sizeof(Triple)*3*nVars*nVals);

	init<<<1,1>>>(dLastValuesMem,dVarsMem,dQueueMem);
}





















