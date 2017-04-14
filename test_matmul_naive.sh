#!/bin/bash

# compiled command:
# icc -mavx -O3 -fopenmp -o matmul_naive matmul_naive.c
cat /dev/null > matmul_naive.csv
echo "i" >> matmul_naive.csv
for i in 1 2 3
do
	./matmul_naive 2000 0 2>> matmul_naive.csv
done

echo "ii" >> matmul_naive.csv
for i in 1 2 4 8 16
do
	for j in 1 2 3
	do
		set OMP_NUM_THREADS $i
		./matmul_naive 2000 2 2>> matmul_naive.csv
	done
done

echo "iii" >> matmul_naive.csv
for i in 128 256 512 1024 2048 4096
do
	for j in 1 2 3
	do
		./matmul_naive $i 1 2>> matmul_naive.csv
	done
done
