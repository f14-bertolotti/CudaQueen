#pragma once
#include <stdio.h>

struct Triple{
	int var;	//variable propagated
	int val;	//value propagated
	int cs;		//constraint used for propagation
};


struct TripleQueue{
	int nVars;
	int nVals;
	int count;		//number of element in queue
	int size;		//number of element

	bool dbg;		//if on is verbose

	Triple* q;

	__device__ TripleQueue(int,int); 	//initialize and allcate

	__device__ void add(int,int,int);	//add a Triple at the end
	__device__ void pop();				//delete last element
	__device__ Triple* front();			//return last element
	__device__ void resize();			//resize if needed
	__device__ bool empty();			//return true if empty
	__device__ void print();			//print

	__device__ ~TripleQueue();			//free memory
};

////////////////////////////////////////////////////////////////////

__device__ TripleQueue::TripleQueue(int nvr,int nvl):
		nVars(nvr),nVals(nvl),size(nvr*nvr),count(0){
	q = (Triple*)malloc(size*sizeof(Triple));
}

////////////////////////////////////////////////////////////////////

__device__ void TripleQueue::add(int var, int val, int cs){
	if(count==size)resize();
	
	q[count].var = var;
	q[count].cs = cs;
	q[count].val = val;

	++count; 
}

////////////////////////////////////////////////////////////////////

__device__ void TripleQueue::pop(){
	if(count == 0){
		printf("Error::TripleQueue::pop::EMPTY QUEUE\n");
		return;
	}
	--count;
}

////////////////////////////////////////////////////////////////////

__device__ Triple* TripleQueue::front(){
	if(count == 0){
		printf("Error::TripleQueue::front::EMPTY QUEUE\n");
		return NULL;
	}
	return &q[count-1];
}

////////////////////////////////////////////////////////////////////

__device__ void TripleQueue::resize(){
	Triple* tq = (Triple*)malloc(2*size*sizeof(Triple));
	for(int i = 0; i < size; ++i)
		tq[i]=q[i];
	
	size = 2*size;
	free(q);
	q = tq;
}

////////////////////////////////////////////////////////////////////

__device__ bool TripleQueue::empty(){
	if(count == 0)return true;
	return false;
}

////////////////////////////////////////////////////////////////////

__device__ void TripleQueue::print(){
	for(int i = 0; i < count; ++i)
		if(q[i].cs!=5)printf("(%d,%d,%d)\n",q[i].var,q[i].val,q[i].cs);
		else printf("\033[35m(%d,%d,%d)\033[0m\n",q[i].var,q[i].val,q[i].cs);
}

////////////////////////////////////////////////////////////////////

__device__ TripleQueue::~TripleQueue(){
	free(q);
}















