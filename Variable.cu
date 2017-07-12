#pragma once
#include <stdio.h>

struct Variable{
	int ground;		//if variable domain is ground
	int changed;	//if variable domain is changed
	int failed;		//if variable domain is empty
	int size;		//domain size

	bool dbg;		//if on is verbose

	int *domain;

__device__ Variable(int);	//inizializza la variabile
							//ne alloca la memoria


__device__ void assign(int);		//assegna la variabile
__device__ void undoAssign(int);	//disfa l'assegnamento
__device__ void addTo(int,int);		//aggiunge ad un elemento delta

__device__ void checkGround();		
__device__ void checkFailed();

__device__ void print(int);	//stampa with modes

__device__ ~Variable();		//dealloca la memoria

};


//////////////////////////////////////////////////////////////////////////////////////////

__device__ Variable::Variable(int sz):size(sz){
	domain = (int*)malloc(size*sizeof(int));
	for(int i = 0; i < size; ++i)
		domain[i]=1;
	ground  = -1;
	changed = -1;
	failed  = -1;
	dbg = false;
}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ void Variable::assign(int val){
	if(val < 0 || val >= size){
		printf("Error::Variable::assign::ASSIGNMENT OUT OF BOUND\n");
		return;
	}

	if(failed == 1){
		printf("Error::Variable::assign::VARIABLE ALREADY FAILED\n");
		return;
	}

	if(domain[val]<=0){
		printf("Error::Variable::assign::VALUE NOT IN DOMAIN\n");
		return;
	}


	if(ground >= 0 && val != ground){
		printf("Error::Variable::assign::VARIABLE ALREADY GROUND\n");
		return;
	}

	for(int i = 0; i < size; ++i){
		if(i != val)--domain[i];
	}

	ground = val;

}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ void Variable::undoAssign(int val){
	if(val < 0 || val >= size){
		printf("Error::Variable::undoAssign::UNDOASSIGNMENT OUT OF BOUND\n");
		return;
	}

	if(ground == -1){
		printf("Error::Variable::undoAssign::UNDOING NOT GROUND VAR\n");
	}

	for(int i = 0; i < size; ++i){
		if(i != val)++domain[i];
	}

	checkGround();

}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ void Variable::addTo(int val,int delta){
	if(val < 0 || val >= size){
		printf("Error::Variable::addTo::ADDING OUT OF BOUND\n");
		return;
	}
	
	if(domain[val] > 0 && domain[val] + delta <= 0) changed = 1;

	domain[val]+=delta;

	checkGround();
	checkFailed();
	
}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ void Variable::checkGround(){
	int sum = 0;
	for(int i = 0; i < size; ++i){
		if(domain[i]==1){
			++sum;
			ground = i;
		}
	}
	if(sum != 1) ground = -1;

}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ void Variable::checkFailed(){
	for(int i = 0; i < size; ++i)
		if(domain[i]==1){
			failed = -1;
			return;
		}
	failed = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////
#define PRINTC(c,f,s) printf ("\033[%dm" f "\033[0m", 30 + c, s)
__device__ void Variable::print(int mode){

	switch(mode){
		case 0:{
			for(int i = 0; i < size; ++i)
				printf("%d ",domain[i]);

			printf(" ::: grd:%d chg:%d fld:%d sz:%d\n",ground,changed,failed,size);
		}break;
		case 1:{
			for(int i = 0; i < size; ++i)
				if(domain[i]==1)printf("%d ",1);
				else printf("%d ", 0);

			printf(" ::: grd:%d chg:%d fld:%d sz:%d\n",ground,changed,failed,size);
		}break;
		case 2:{
			for(int i = 0; i < size; ++i){
				if(domain[i]==1)printf("%d ",domain[i]);
				if(domain[i]==0)PRINTC (4, "%d ", domain[i]);
				if(domain[i]<0)PRINTC (1, "%d ", -domain[i]);
			}
	
			printf(" ::: grd:%d chg:%d fld:%d sz:%d\n",ground,changed,failed,size);
		}break;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////

__device__ Variable::~Variable(){
	if(dbg)printf("msg::Variable::~Variable::FREE VAR\n");
	free(domain);
}

///////////////////////////////////////////////////////////////////////////////////////////


















