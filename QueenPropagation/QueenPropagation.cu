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

	__device__ int static inline parallelForwardPropagation(DeviceVariableCollection&,int,int,cudaStream_t&);	//forward, uses parallelPropagation code 5
	__device__ int static inline parallelForwardPropagation(DeviceVariableCollection&,int,int);	//forward, uses parallelPropagation code 5
	__device__ int static inline parallelForwardPropagation2(DeviceVariableCollection&,int,int);
	__device__ int static inline parallelForwardPropagation2(DeviceVariableCollection&,int,int,cudaStream_t&);
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

	if (vc.isFailed()) return -1;

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void externParallelPropagation2(DeviceVariableCollection& vc, int var, int val){

	int index = threadIdx.x +blockIdx.x*blockDim.x;
	int columnIndex = int((index % (vc.nQueen * vc.nQueen))%vc.nQueen);
	int rowIndex = int(((index % (vc.nQueen * vc.nQueen))/vc.nQueen) % vc.nQueen);

	if(index < vc.nQueen*vc.nQueen){
		if(rowIndex != var && val == columnIndex){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
		
		if(rowIndex != var && columnIndex == rowIndex && columnIndex+val-var < vc.nQueen && columnIndex+val-var >= 0){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+val-var],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
		
		if(rowIndex != var && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-val)+var < vc.nQueen && columnIndex-(vc.nQueen-val)+var >= 0){

			int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-val)+var],-1);
			if(old == 1){
				vc.deviceVariable[rowIndex].changed = 1;
			}

		}
	}
	__syncthreads();

	if(index == 0){
		int old = atomicAdd(&vc.deviceQueue.count,1);
		vc.deviceQueue.q[old].var = var;
		vc.deviceQueue.q[old].val = val;
		vc.deviceQueue.q[old].cs = 6;

	}

	if(index >= vc.nQueen && index < vc.nQueen*2)
		vc.deviceVariable[index-vc.nQueen].checkFailed();

	if(index >= vc.nQueen*2 && index < vc.nQueen*3)
		vc.deviceVariable[index-vc.nQueen*2].checkGround();

	bool ch = false;

	__syncthreads();


	do{
		
		__syncthreads();
		
		ch=false;
		
		for(int i = var+1; i < vc.nQueen; ++i){


			if(vc.deviceVariable[i].changed == 1){

				__syncthreads();

				if(vc.deviceVariable[i].ground>=0){

					__syncthreads();

					if(index < vc.nQueen*vc.nQueen){
						if(rowIndex != i && vc.deviceVariable[i].ground == columnIndex){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
						
						if(rowIndex != i && columnIndex == rowIndex && columnIndex+vc.deviceVariable[i].ground-i < vc.nQueen && columnIndex+vc.deviceVariable[i].ground-i >= 0){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex+vc.deviceVariable[i].ground-i],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
						
						if(rowIndex != i && vc.nQueen-columnIndex == rowIndex && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i < vc.nQueen && columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i >= 0){

							int old = atomicAdd(&vc.deviceVariable[rowIndex].domain[columnIndex-(vc.nQueen-vc.deviceVariable[i].ground)+i],-1);
							if(old == 1){
								vc.deviceVariable[rowIndex].changed = 1;
							}

						}
					}

					__syncthreads();

					if(index == 0){
						int old = atomicAdd(&vc.deviceQueue.count,1);
						vc.deviceQueue.q[old].var = i;
						vc.deviceQueue.q[old].val = vc.deviceVariable[i].ground;
						vc.deviceQueue.q[old].cs = 6;

					}

					if(index >= vc.nQueen && index < vc.nQueen*2)
						vc.deviceVariable[index-vc.nQueen].checkFailed();

					if(index >= vc.nQueen*2 && index < vc.nQueen*3)
						vc.deviceVariable[index-vc.nQueen*2].checkGround();

					ch = true;
					__syncthreads();
				}

				__syncthreads();
				vc.deviceVariable[i].changed=-1;
			}
		}
		__syncthreads();

	}while(ch);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::parallelForwardPropagation2(DeviceVariableCollection& vc, int var, int val){



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

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelForwardPropagation::STREAM CREATION");

	externParallelPropagation2<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,var,val);

	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelForwardPropagation::EXTERN FORWARD PROPAGATION CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelForwardPropagation::STREAM DESTRUCTION");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelForwardPropagation::SYNCH");


	if(vc.isFailed()){
		vc.deviceQueue.add(var,val,5);
		return 1;
	}

	vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::parallelForwardPropagation2(DeviceVariableCollection& vc, int var, int val, cudaStream_t& s){



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

	externParallelPropagation2<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,var,val);

	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelForwardPropagation::EXTERN FORWARD PROPAGATION CALL");
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelForwardPropagation::SYNCH");

	vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void externCheckAll(DeviceVariableCollection& vc){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < vc.nQueen) vc.deviceVariable[index].checkFailed();
	if(index >= vc.nQueen) vc.deviceVariable[index-vc.nQueen].checkGround();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void externPropagation2(DeviceVariableCollection& vc, int var, int val, int nQueen,int delta){

	int col = int((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))%nQueen);
	int row = int(((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))/nQueen) % nQueen);

	if(row != var && val == col){
		atomicAdd(&vc.deviceVariable[row].domain[col],1);
	}
	
	if(row != var && col == row && col+val-var < vc.nQueen && col+val-var >= 0){
		atomicAdd(&vc.deviceVariable[row].domain[col+val-var],1);
	}
	
	if(row != var && vc.nQueen-col == row && col-(vc.nQueen-val)+var < vc.nQueen && col-(vc.nQueen-val)+var >= 0){
		atomicAdd(&vc.deviceVariable[row].domain[col-(vc.nQueen-val)+var],1);
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
		externPropagation2<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,vc.nQueen,+1);
		ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelUndoForwardPropagation::EXTERN PROPAGATION CALL");
		ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM DESTRUCTION");

		vc.deviceQueue.pop();
		if(vc.deviceQueue.empty())break;
	}
	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM SYNCH");

	cudaStream_t s;
	ErrorChecking::deviceErrorCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM CREATION");
	externCheckAll<<<1,vc.nQueen*2,0,s>>>(vc);
	ErrorChecking::deviceErrorCheck(cudaPeekAtLastError(),"DeviceQueenPropagation::parallelUndoForwardPropagation::EXTERN PROPAGATION CALL");
	ErrorChecking::deviceErrorCheck(cudaStreamDestroy(s),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM DESTRUCTION");

	vc.deviceVariable[t1].undoAssign(t2);

	ErrorChecking::deviceErrorCheck(cudaDeviceSynchronize(),"DeviceQueenPropagation::parallelUndoForwardPropagation::STREAM SYNCH");

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
