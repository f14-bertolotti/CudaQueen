#include "../WorkSet/WorkSet.cu"
#include "../QueenConstraints/QueenConstraints.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ DeviceWorkSet deviceWorkSet;
__device__ DeviceQueenConstraints deviceQueenConstraints;
__device__ DeviceQueenPropagation deviceQueenPropagation;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariableCollection*,
					 DeviceVariable*,
					 int*,int*,Triple*,int,int);
__global__ void test();

/////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){

	HostWorkSet hostWorkSet(5,10);
	init<<<1,1>>>(hostWorkSet.deviceVariableCollection,
				  hostWorkSet.deviceVariable,
				  hostWorkSet.variablesMem,
				  hostWorkSet.lastValuesMem,
				  hostWorkSet.tripleQueueMem,
				  hostWorkSet.nQueen,
				  hostWorkSet.nVariableCollection);
	cudaDeviceSynchronize();

	test<<<1,1>>>();

	return 0;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init(DeviceVariableCollection* deviceVariableCollection,
					 DeviceVariable* deviceVariable,
					 int* variablesMem, int* lastValuesMem,
					 Triple* tripleQueueMem, int nQueen, int nVariableCollection){

	deviceWorkSet.init(deviceVariableCollection,
					   deviceVariable,
					   variablesMem,
					   lastValuesMem,
					   tripleQueueMem,
					   nQueen,
					   nVariableCollection);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void test(){

    deviceWorkSet.expand(0,0);
    deviceWorkSet.expand(1,1);
	deviceWorkSet.print();

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
------[0]------
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5

------[1]------
[1] ::: 1 0 0 0 0  ::: grd:0 chd:-1 fld:-1 sz:5
[0] ::: 0 0 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 0 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 1 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 1 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
(0,0,6)
(0,0,5)

------[2]------
[2] ::: 0 1 0 0 0  ::: grd:1 chd:-1 fld:-1 sz:5
[0] ::: 0 0 0 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
(0,1,6)
(0,1,5)

------[3]------
[3] ::: 0 0 1 0 0  ::: grd:2 chd:-1 fld:-1 sz:5
[0] ::: 1 0 0 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 0 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 0 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 0 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
(0,2,6)
(0,2,5)

------[4]------
[4] ::: 0 0 0 1 0  ::: grd:3 chd:-1 fld:-1 sz:5
[0] ::: 1 1 0 0 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 1 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 0 1  ::: grd:-1 chd:-1 fld:-1 sz:5
(0,3,6)
(0,3,5)

------[5]------
[5] ::: 0 0 0 0 1  ::: grd:4 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 0 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 0 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 0 1 1 1 0  ::: grd:-1 chd:-1 fld:-1 sz:5
(0,4,6)
(0,4,5)

------[6]------
[1] ::: 1 2 2 1 1  ::: grd:0 chd:-1 fld:-1 sz:5
[3] ::: 2 2 1 3 1  ::: grd:2 chd:-1 fld:-1 sz:5
[0] ::: 1 2 2 1 1  ::: grd:4 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 2 2  ::: grd:1 chd:-1 fld:-1 sz:5
[0] ::: 1 0 2 1 1  ::: grd:3 chd:-1 fld:-1 sz:5
(0,0,6)
(0,0,5)
(1,2,6)
(2,4,6)
(3,1,6)
(4,3,6)
(1,2,5)

------[7]------
[1] ::: 1 2 2 1 1  ::: grd:0 chd:-1 fld:-1 sz:5
[4] ::: 2 2 3 1 1  ::: grd:3 chd:-1 fld:-1 sz:5
[0] ::: 1 1 2 1 2  ::: grd:1 chd:-1 fld:-1 sz:5
[0] ::: 1 2 1 2 1  ::: grd:4 chd:-1 fld:-1 sz:5
[0] ::: 1 0 1 2 1  ::: grd:2 chd:-1 fld:-1 sz:5
(0,0,6)
(0,0,5)
(1,3,6)
(2,1,6)
(3,4,6)
(4,2,6)
(1,3,5)

------[8]------
[1] ::: 1 1 1 1 1  ::: grd:0 chd:-1 fld:-1 sz:5
[5] ::: 2 2 2 0 1  ::: grd:4 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 0 1  ::: grd:1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 2 1 0  ::: grd:-1 chd:-1 fld:1 sz:5
[0] ::: 0 1 1 0 1  ::: grd:2 chd:-1 fld:-1 sz:5
(0,0,6)
(0,0,5)
(1,4,6)
(2,1,6)
(4,2,6)
(1,4,5)

------[9]------
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5
[0] ::: 1 1 1 1 1  ::: grd:-1 chd:-1 fld:-1 sz:5

count : 9

*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////