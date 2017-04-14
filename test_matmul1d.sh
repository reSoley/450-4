#!/bin/bash

#compiled with
#mpicc -O3 -o matmul1d matmul1d.c
echo "time,comm,comm/time"
for i in 1 2 4 8 16 32 64
do
	echo "# procs $i"
	for k in 1 2 3
	do
		ibrun -np $i ./matmul1d 2000
	done
done
