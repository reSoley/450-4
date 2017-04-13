#!/bin/bash

for i in 1 2 4 8 16 32 64
do
	for k in 1 2 3
	do
		ibrun -np $i ./matmul1d 2000
	done
done
