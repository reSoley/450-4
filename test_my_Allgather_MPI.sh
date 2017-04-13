#!/bin/bash

for i in 1 2 4 8 16 32 64
do
	for j in 1 10 100 1000 10000
	do
		mpirun -np $i ./my_Allgather $j
	done
done
