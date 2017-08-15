#!/bin/bash


echo "NQUEEN SOLUTIONS NODES BLOCKUSED QUEUEUSED MAXBLOCK MAXQUEUE L1 L2 TIME" > 1BlockSimple;


START=0;
QUEEN=12;
BLOCK=60000;
QUEUE=60000;

for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 4 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 5 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 6 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 7 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 8 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 9 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 10 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 11 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 12 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 13 -l 2 -p -m 0 >> 1BlockSimple; done;
for (( i=0; i<5; i++ )); do ./RCUDAqueen0.7.3 -n 14 -l 2 -p -m 0 >> 1BlockSimple; done;


shutdown -h now;
