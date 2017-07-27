#pragma once
#include "../VariableCollection/VariableCollection.cu"
#include "../ErrorChecking/ErrorChecking.cu"

struct DeviceQueenPropagation{

	//////////////////////////////////////SINGLE THREAD//////////////////////////////////////

	__device__ int static inline nextAssign(DeviceVariableCollection&,int);		//assign next value not already tried
																	//returns assigned value

	__device__ int static inline allDifferent(DeviceVariableCollection&,int,int,int);		//propagate for all different constraint code 3
	__device__ int static inline diagDifferent(DeviceVariableCollection&,int,int,int);	//propagate for diag constraint code 4

	__device__ int static inline forwardPropagation(DeviceVariableCollection&,int,int);	//csp forward propagation code 5
	__device__ int static inline undoForwardPropagation(DeviceVariableCollection&);		//csp undo forward propagation

	//////////////////////////////////////MULTI THREAD//////////////////////////////////////

	__device__ int static inline parallelPropagation(DeviceVariableCollection&,int,int,int); 		//propagation multithread code 2
	__device__ int static inline parallelForwardPropagation(DeviceVariableCollection&,int,int);	//forward, uses parallelPropagation code 5
	__device__ int static inline parallelUndoForwardPropagation(DeviceVariableCollection&);		//csp undo forward propagation

};

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::nextAssign(DeviceVariableCollection& vc, int var){

	if(var < 0 || var >= vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::nextAssign::VAR OUT OF BOUND");
		return -1;
	}

	if(vc.lastValues[var] >= vc.nQueen){
		ErrorChecking::deviceMessage("Warn::DeviceQueenPropagation::nextAssign::VALUE OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].failed == 1){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::nextAssign::VAR ALREADY FAILED");
		return -1;
	}

	int next;
	for(next = vc.lastValues[var];next<vc.nQueen;++next)
		if(vc.deviceVariable[var].domain[next]==1){
			vc.lastValues[var]=next+1;
			vc.deviceVariable[var].assign(next);
			return next;
		}

	ErrorChecking::deviceMessage("Warn::DeviceQueenPropagation::nextAssign::NEXTVALUE NOT FOUND");

	return -1;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::allDifferent(DeviceVariableCollection& vc, int var, int val, int delta){

	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::allDifferent::OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::QueenPropagation::allDifferent::VARIABLE NOT GROUND");
		return -1;
	}
	
	for(int i = 0; i < vc.nQueen; ++i)
		if(i != var){
			vc.deviceVariable[i].addTo(val,delta);

		}
	
	if(delta < 0)vc.deviceQueue.add(var,val,3);

	return 0;	

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::diagDifferent(DeviceVariableCollection& vc, int var, int val, int delta){

	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::diagDifferent::OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::diagDifferent::VARIABLE NOT GROUND");
		return -1;
	}

	int i=var+1,j=val+1;
	while(i<vc.nQueen && j<vc.nQueen){
		vc.deviceVariable[i].addTo(j,delta);
		++i;++j;
	}

	i=var-1,j=val-1;
	while(i>=0 && j>=0){
		vc.deviceVariable[i].addTo(j,delta);
		--i;--j;
	}

	i=var-1,j=val+1;
	while(i>=0 && j<vc.nQueen){
		vc.deviceVariable[i].addTo(j,delta);
		--i;++j;
	}

	i=var+1,j=val-1;
	while(i<vc.nQueen && j>=0){
		vc.deviceVariable[i].addTo(j,delta);
		++i;--j;
	}

	if(delta < 0)vc.deviceQueue.add(var,val,4);
	return 0;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::forwardPropagation(DeviceVariableCollection& vc, int var, int val){

	if(var < 0 || var > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::forwardPropagation:: VAR OUT OF BOUND");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::forwardPropagation:: VAL OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::forwardPropagation::VARIABLE NOT GROUND");
		return -1;
	}

	allDifferent(vc,var,val,-1);
	diagDifferent(vc,var,val,-1);

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.deviceVariable[i].changed==1){
				if(vc.deviceVariable[i].ground>=0){
					allDifferent(vc,i,vc.deviceVariable[i].ground,-1);
					diagDifferent(vc,i,vc.deviceVariable[i].ground,-1);
					ch = true;
				}
				vc.deviceVariable[i].changed=-1;
			}
		}
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	return 0;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::undoForwardPropagation(DeviceVariableCollection& vc){

	if(vc.deviceQueue.front()->cs!=5){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::undoForwardPropagation::ERROR IN QUEUE");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::undoForwardPropagation::EMPTY QUEUE");
		return -1;		
	}

	int t1=vc.deviceQueue.front()->var;
	int t2=vc.deviceQueue.front()->val;

	for(int i = t1+1; i < vc.nQueen; ++i)vc.lastValues[i]=0;

	vc.deviceQueue.pop();
	while(vc.deviceQueue.front()->cs!=5){
		switch(vc.deviceQueue.front()->cs){
			case 3:{
				allDifferent(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,+1);	
			}break;
			case 4:{
				diagDifferent(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,+1);	
			}break;
		}
		vc.deviceQueue.pop();

		if(vc.deviceQueue.empty())break;
	}

	vc.deviceVariable[t1].undoAssign(t2);
	return 0;

}

////////////////////////////////////////////////////////////////////////////

__global__ void externPropagation(DeviceVariableCollection& vc, int var, int val, int nQueen,int delta){

	int col = int((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))%nQueen);
	int row = int(((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))/nQueen) % nQueen);

	if(row != var && val == col)
		vc.deviceVariable[row].addTo(col,delta);
	
	
	if(row != var && col == row && col+val-var < nQueen && col+val-var >= 0)
		vc.deviceVariable[row].addTo(col+val-var,delta);
	

	if(row != var && nQueen-col == row && col-(nQueen-val)+var < nQueen && col-(nQueen-val)+var >= 0)
		vc.deviceVariable[row].addTo(col-(nQueen-val)+var,delta);

}

__device__ int inline DeviceQueenPropagation::parallelPropagation(DeviceVariableCollection& vc,int var,int val,int delta){

	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelPropagation::OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::QueenPropagation::parallelPropagation::VARIABLE NOT GROUND");
		return -1;
	}

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelPropagation::STREAM CREATION");
	externPropagation<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,var,val,vc.nQueen,delta);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelPropagation::EXTERN PROPAGATION CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelPropagation::STREAM DESTRUCTION");
	if(delta < 0)vc.deviceQueue.add(var,val,6);
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelPropagation::SYNCH");

	return 0;

}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::parallelForwardPropagation(DeviceVariableCollection& vc, int var, int val){

	if(var < 0 || var > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAR OUT OF BOUND");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VAL OUT OF BOUND");
		return -1;
	}

	if(vc.deviceVariable[var].ground != val){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelForwardPropagation::VARIABLE NOT GROUND");
		return -1;
	}

	parallelPropagation(vc,var,val,-1);

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.deviceVariable[i].changed==1){
				if(vc.deviceVariable[i].ground>=0){
					parallelPropagation(vc,i,vc.deviceVariable[i].ground,-1);
					ch = true;
				}
				vc.deviceVariable[i].changed=-1;
			}
		}
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ int inline DeviceQueenPropagation::parallelUndoForwardPropagation(DeviceVariableCollection& vc){

	if(vc.deviceQueue.front()->cs!=5){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelUndoForwardPropagation::ERROR IN QUEUE");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		ErrorChecking::deviceError("Error::DeviceQueenPropagation::parallelUndoForwardPropagation::EMPTY QUEUE");
		return -1;		
	}

	int t1=vc.deviceQueue.front()->var;
	int t2=vc.deviceQueue.front()->val;

	for(int i = t1+1; i < vc.nQueen; ++i)vc.lastValues[i]=0;

	vc.deviceQueue.pop();
	while(vc.deviceQueue.front()->cs!=5){
		cudaStream_t s;
		ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM CREATION");
		externPropagation<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,vc.nQueen,+1);
		ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelUndoForwardPropagation::EXTERN PROPAGATION CALL");
		ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM DESTRUCTION");
		ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM SYNCH");
		vc.deviceQueue.pop();
		if(vc.deviceQueue.empty())break;
	}

	vc.deviceVariable[t1].undoAssign(t2);

	return 0;
}

////////////////////////////////////////////////////////////////////////////

