#!/bin/bash

for i in 1 2 4 8 16 32 64
do
	for j in 5000 10000
	do
		for k in 10 100 1000
		do
			mpirun -np $i ./gameoflife_MPI $j $k
		done
	done
done
