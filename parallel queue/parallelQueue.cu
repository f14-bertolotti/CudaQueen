
#pragma once
#include <stdio.h>

template <typename T>
struct parallelQueue{
	int lockCount;								//lock on count variable
	int count;									//number of element in queue
	int size;									//max number of element(fixed)
	bool dbg;
	int* lockReading;
	T* queue;

	__device__ parallelQueue();					//do nothing
	__device__ parallelQueue(T*,int*,int,int);	//initialize
	__device__ void init(T*,int*,int,int);		//initialize

	__device__ int add(T&);						//add an element, -1 if fail
	__device__ int pop();						//delete last element , -1 if fail
	__device__ int frontAndPop(T&);				//returns last and delete last element, -1 if fail
	__device__ bool empty();					//return true if empty

	__device__ void print();					//print
	__device__ void printLocks();				//do nothing
												//prints are not locked

	__device__ ~parallelQueue();				//do nothing
};

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ parallelQueue<T>::parallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ parallelQueue<T>::parallelQueue(T* q, int* lr, int cn, int sz):lockCount(0),
	count(cn),size(sz),dbg(true),lockReading(lr),queue(q){}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ void parallelQueue<T>::init(T* q, int* lr, int cn, int sz){
	lockCount = 0;
	count = cn;
	size = sz;
	queue = q;
	lockReading = lr;
	dbg = true;
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ int parallelQueue<T>::add(T& element){

	int temp = 0;
	while(atomicCAS(&lockCount,0,1)==1){}
	if(count == size){
		printf("Warn::parallelQueue::add::NOT ENOUGH SPACE\n");
		return -1;
	}
	temp = count;
	++count;
	lockCount = 0;

	while(atomicCAS(&lockReading[temp],0,1)==1){}
	queue[temp].copy(element);
	lockReading[temp] = 0;
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ int parallelQueue<T>::frontAndPop(T& element){

	while(atomicCAS(&lockCount,0,1)==1){}
	if(count > size || count <= 0){
		if(dbg)
			printf("Warn::parallelQueue::frontAndPop::OUT OF BOUNDS\n");
		lockCount = 0;
		return -1;
	}
	while(atomicCAS(&lockReading[count-1],0,1)==1){}

	count--;

	element.copy(queue[count]);
	lockReading[count] = 0;
	lockCount = 0;

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ int parallelQueue<T>::pop(){
	while(atomicCAS(&lockCount,0,1)==1){}
	if(count <= 0){
		printf("Warn::parallelQueue::pop::EMPTY QUEUE\n");
		lockCount = 0;
		return -1;
	}
	while(atomicCAS(&(lockReading[count-1]),0,1)==1){}
	--count;
	lockReading[count] = 0;
	lockCount = 0;

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ bool parallelQueue<T>::empty(){
	int temp = 0;
	while(atomicCAS(&lockCount,0,1)==1){}
	temp = count;
	lockCount = 0;
	return temp==0 ? true : false;
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ void parallelQueue<T>::print(){

	for(int i = 0; i < count; ++i) {
		printf("index: %d - ", i);
		queue[i].print();
	}

	printf("count:%d\n",count);
	printf("size:%d\n",size);
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ void parallelQueue<T>::printLocks(){

	printf("lock count: %d\n",lockCount);
	printf("locks reading:\n");
	for (int i = 0; i < size; ++i){
		if(i%100==0 && i != 0)printf("\n");
		printf("%d",lockReading[i]);
	}
	printf("\n");
}


//////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ parallelQueue<T>::~parallelQueue(){}

//////////////////////////////////////////////////////////////////////////////////////////////
