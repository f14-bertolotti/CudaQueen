#include <stdio.h>
#include "./Variable.cu"
#include "./VariableCollection.cu"
#include "./QueenConstraints.cu"
#include "./QueenPropagation.cu"
#include "./TripleQueue.cu"


__device__ int nVars=8;
__device__ int nVals=8;

__global__ void test(){

	QueenConstraints qc(nVars,nVals);
	VariableCollection vc(nVars,nVals);
	QueenPropagation pr(vc,nVars,nVals);


	int level = 0;
//	int loop = 0;
	int val = 0;
	int nSols = 0;
	bool done = false;
	do{
//		printf("level::%d\n",level);
		if(level == nVars){
			if(vc.isSolution()){
				++nSols;
/*				printf("----SOLUTION----\n");
				vc.print(2);
				printf("----------------\n");*/
			}
			pr.undoForwardPropagation();
			--level;			
		}else{

			val = pr.nextAssign(level);
			if(val == -1){
				if(level == 0) done = true;
				else{
					pr.undoForwardPropagation();
					--level;
				}
			}else{
				pr.forwardPropagation(level,val);
				if(vc.isFailed()){
/*					printf("----FAILED----\n");
					vc.print(2);
					printf("--------------\n");*/
					pr.undoForwardPropagation();
					--level;
				}
				++level;
			}
		}
/*		vc.print(2);
		++loop;
		if(loop >= 5000){
			printf("ENDED BY LOOP LIMIT\n");
			return;
		}*/
	}while(!done);

	printf("\033[32mSOLUTIONS FOUND = %d\033[0m\n",nSols);
}

int main(){


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

