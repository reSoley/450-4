#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

static double timer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

int my_Allgather(int *sendbuf, int n, int nprocs, int rank, int *recvbuf) {
	/* Recursive doubling-based code goes here */
	MPI_Status status;
	int swap_rank, i = 1;

	/* As base case, place sendbuf in proper location in recvbuf */
	memcpy(recvbuf+(n*rank), sendbuf, n*i*sizeof(int));

	while ((i < nprocs) && (rank % i == 0)) {
		/* Generate swap parter using XOR */
		swap_rank = rank ^ i;

		/* Swap array chunks with the process i processes away */
    		MPI_Sendrecv(recvbuf+(n*rank), n*i, MPI_INT, swap_rank, 123, recvbuf+(n*swap_rank),
					n*i, MPI_INT, swap_rank, 123, MPI_COMM_WORLD, &status);

		/* Left shift i to next power of 2 */
		i = i << 1;
	}

	/* Broadcast current master array from root to all processes */
	MPI_Bcast(recvbuf, n*nprocs, MPI_INT, 0, MPI_COMM_WORLD);

	return 0;
}

int main(int argc, char **argv) {

	int rank, nprocs;
	int i;

	/* Initialize MPI Environment */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		printf("If nprocs (%d) is not a power of 2 program will abort\n", nprocs);
	}

	/* Change the input size here */
	int n;
	if (argc == 2) {
		n = atoi(argv[1]);
	} else {
		n = 4;
	}
	if (rank == 0) {
		printf("n: %d\n", n);
    	}

	int *sendbuf;
	int *recvbuf1;
	int *recvbuf2;
 
	sendbuf  = (int *) malloc(n*sizeof(int));
	assert(sendbuf != 0);

	for (i=0; i<n; i++) {
	sendbuf[i] = (rank+1);
	}

	recvbuf1 = (int *) malloc(n*nprocs*sizeof(int));
	assert(recvbuf1 != 0);

	recvbuf2 = (int *) malloc(n*nprocs*sizeof(int));
	assert(recvbuf2 != 0);

	double elt_MPI, elt_my;

	if (rank == 0) {
		elt_MPI = timer();
	}
	MPI_Allgather(sendbuf, n, MPI_INT, recvbuf1, n, MPI_INT, MPI_COMM_WORLD);
	if (rank == 0) {
		elt_MPI = timer() - elt_MPI;
		elt_my = timer();
	}
	my_Allgather(sendbuf, n, nprocs, rank, recvbuf2);
	if (rank == 0) {
		elt_my = timer() - elt_my;
	}
 
	/* Verify that my_Allgather works correctly */
	for (i=0; i<n*nprocs; i++) {
		assert(recvbuf1[i] == recvbuf2[i]);
		// printf("mpi: %d, my: %d\n", recvbuf1[i], recvbuf2[i]);
	}
	if (rank == 0) {
		printf("Verification complete\n");
		printf("\tMPI_Allgather: %3.5lf s\n\tmy_Allgather: %3.5lf s\n\n", elt_MPI, elt_my);
	}

	free(sendbuf); free(recvbuf1); free(recvbuf2);
	
	/* Terminate MPI environment */
	MPI_Finalize();

	return 0;
}
