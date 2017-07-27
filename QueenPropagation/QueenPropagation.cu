#pragma once
#include "../VariableCollection/VariableCollection.cu"

struct DeviceQueenPropagation{

	__device__ DeviceQueenPropagation();			//do nothing
	__device__ ~DeviceQueenPropagation();			//do nothing


	//////////////////////////////////////SINGLE THREAD//////////////////////////////////////

	__device__ int nextAssign(DeviceVariableCollection&,int);		//assign next value not already tried
																	//returns assigned value

	__device__ int allDifferent(DeviceVariableCollection&,int,int,int);		//propagate for all different constraint code 3
	__device__ int diagDifferent(DeviceVariableCollection&,int,int,int);	//propagate for diag constraint code 4

	__device__ int forwardPropagation(DeviceVariableCollection&,int,int);	//csp forward propagation code 5
	__device__ int undoForwardPropagation(DeviceVariableCollection&);		//csp undo forward propagation

	//////////////////////////////////////MULTI THREAD//////////////////////////////////////

	__device__ int parallelPropagation(DeviceVariableCollection&,int,int,int); 		//propagation multithread code 2
	__device__ int parallelForwardPropagation(DeviceVariableCollection&,int,int);	//forward, uses parallelPropagation code 5
	__device__ int parallelUndoForwardPropagation(DeviceVariableCollection&);		//csp undo forward propagation

};

////////////////////////////////////////////////////////////////////////////

__device__ DeviceQueenPropagation::DeviceQueenPropagation(){}


////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::nextAssign(DeviceVariableCollection& vc, int var){

	if(var < 0 || var >= vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::nextAssign::VAR OUT OF BOUND\n");
		return -1;
	}

	if(vc.lastValues[var] >= vc.nQueen){
		if(vc.dbg)printf("\033[34mWarn\033[0m::DeviceQueenPropagation::nextAssign::VALUE OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].failed == 1){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::nextAssign::VAR ALREADY FAILED\n");
		return -1;
	}

	int next;
	for(next = vc.lastValues[var];next<vc.nQueen;++next)
		if(vc.variables[var].domain[next]==1){
			vc.lastValues[var]=next+1;
			vc.variables[var].assign(next);
			return next;
		}

	if(vc.dbg)printf("\033[34mWarn\033[0m::DeviceQueenPropagation::nextAssign::NEXTVALUE NOT FOUND\n");
	return -1;
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::allDifferent(DeviceVariableCollection& vc, int var, int val, int delta){
	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::allDifferent::OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].ground != val){
		printf("\033[31mError\033[0m::QueenPropagation::allDifferent::VARIABLE NOT GROUND\n");
		return -1;
	}
	
	for(int i = 0; i < vc.nQueen; ++i)
		if(i != var){
			vc.variables[i].addTo(val,delta);

		}
	
	if(delta < 0)vc.deviceQueue.add(var,val,3);

	return 0;	
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::diagDifferent(DeviceVariableCollection& vc, int var, int val, int delta){
	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::diagDifferent::OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].ground != val){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::diagDifferent::VARIABLE NOT GROUND\n");
		return -1;
	}

	int i=var+1,j=val+1;
	while(i<vc.nQueen && j<vc.nQueen){
		vc.variables[i].addTo(j,delta);
		++i;++j;
	}

	i=var-1,j=val-1;
	while(i>=0 && j>=0){
		vc.variables[i].addTo(j,delta);
		--i;--j;
	}

	i=var-1,j=val+1;
	while(i>=0 && j<vc.nQueen){
		vc.variables[i].addTo(j,delta);
		--i;++j;
	}

	i=var+1,j=val-1;
	while(i<vc.nQueen && j>=0){
		vc.variables[i].addTo(j,delta);
		++i;--j;
	}

	if(delta < 0)vc.deviceQueue.add(var,val,4);
	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::forwardPropagation(DeviceVariableCollection& vc, int var, int val){

	if(var < 0 || var > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::forwardPropagation:: VAR OUT OF BOUND\n");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::forwardPropagation:: VAL OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].ground != val){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::forwardPropagation::VARIABLE NOT GROUND\n");
		return -1;
	}

	allDifferent(vc,var,val,-1);
	diagDifferent(vc,var,val,-1);

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.variables[i].changed==1){
				if(vc.variables[i].ground>=0){
					allDifferent(vc,i,vc.variables[i].ground,-1);
					diagDifferent(vc,i,vc.variables[i].ground,-1);
					ch = true;
				}
				vc.variables[i].changed=-1;
			}
		}
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::undoForwardPropagation(DeviceVariableCollection& vc){
	if(vc.deviceQueue.front()->cs!=5){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::undoForwardPropagation::ERROR IN QUEUE\n");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::undoForwardPropagation::EMPTY QUEUE\n");
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

	vc.variables[t1].undoAssign(t2);
	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ DeviceQueenPropagation::~DeviceQueenPropagation(){}

////////////////////////////////////////////////////////////////////////////

__global__ void externPropagation(DeviceVariableCollection& vc, int var, int val, int nQueen,int delta){
	int col = int((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))%nQueen);
	int row = int(((threadIdx.x + blockIdx.x * blockDim.x % (nQueen * nQueen))/nQueen) % nQueen);

	if(row != var && val == col)
		vc.variables[row].addTo(col,delta);
	
	
	if(row != var && col == row && col+val-var < nQueen && col+val-var >= 0)
		vc.variables[row].addTo(col+val-var,delta);
	

	if(row != var && nQueen-col == row && col-(nQueen-val)+var < nQueen && col-(nQueen-val)+var >= 0)
		vc.variables[row].addTo(col-(nQueen-val)+var,delta);
}

__device__ int DeviceQueenPropagation::parallelPropagation(DeviceVariableCollection& vc,int var,int val,int delta){
	if(var < 0 || var > vc.nQueen || val < 0 || val > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelPropagation::OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].ground != val){
		printf("\033[31mError\033[0m::QueenPropagation::parallelPropagation::VARIABLE NOT GROUND\n");
		return -1;
	}
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	externPropagation<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,var,val,vc.nQueen,delta);
	cudaStreamDestroy(s);
	if(delta < 0)vc.deviceQueue.add(var,val,6);
	cudaDeviceSynchronize();

	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::parallelForwardPropagation(DeviceVariableCollection& vc, int var, int val){

	if(var < 0 || var > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelForwardPropagation:: VAR OUT OF BOUND\n");
		return -1;
	}

	if(val < 0 || val > vc.nQueen){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelForwardPropagation:: VAL OUT OF BOUND\n");
		return -1;
	}

	if(vc.variables[var].ground != val){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelForwardPropagation::VARIABLE NOT GROUND\n");
		return -1;
	}

	parallelPropagation(vc,var,val,-1);

	bool ch = false;
	do{
		ch=false;
		for(int i = 0; i < vc.nQueen; ++i){
			if(vc.variables[i].changed==1){
				if(vc.variables[i].ground>=0){
					parallelPropagation(vc,i,vc.variables[i].ground,-1);
					ch = true;
				}
				vc.variables[i].changed=-1;
			}
		}
	}while(ch);

	vc.deviceQueue.add(var,val,5);

	return 0;
}

////////////////////////////////////////////////////////////////////////////

__device__ int DeviceQueenPropagation::parallelUndoForwardPropagation(DeviceVariableCollection& vc){
	if(vc.deviceQueue.front()->cs!=5){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelUndoForwardPropagation::ERROR IN QUEUE\n");
		return -1;		
	}

	if(vc.deviceQueue.empty()){
		printf("\033[31mError\033[0m::DeviceQueenPropagation::parallelUndoForwardPropagation::EMPTY QUEUE\n");
		return -1;		
	}

	int t1=vc.deviceQueue.front()->var;
	int t2=vc.deviceQueue.front()->val;

	for(int i = t1+1; i < vc.nQueen; ++i)vc.lastValues[i]=0;

	vc.deviceQueue.pop();
	while(vc.deviceQueue.front()->cs!=5){
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		externPropagation<<<1,vc.nQueen*vc.nQueen,0,s>>>(vc,vc.deviceQueue.front()->var,vc.deviceQueue.front()->val,vc.nQueen,+1);
		cudaStreamDestroy(s);
		cudaDeviceSynchronize();
		vc.deviceQueue.pop();
		if(vc.deviceQueue.empty())break;
	}

	vc.variables[t1].undoAssign(t2);

	return 0;
}

////////////////////////////////////////////////////////////////////////////

