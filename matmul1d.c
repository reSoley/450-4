#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define USE_MPI 1

#if USE_MPI
#include <mpi.h>
#endif

static double timer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

int main(int argc, char **argv) {

    int rank, num_tasks;

    /* Initialize MPI */
#if USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status stat;
    //printf("Hello world from rank %3d of %3d\n", rank, num_tasks);
#else
    rank = 0;
    num_tasks = 1;
#endif

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "%s <n>\n", argv[0]);
            fprintf(stderr, "Program for parallel dense matrix-matrix multiplication\n");
            fprintf(stderr, "with 1D row partitioning\n");
            fprintf(stderr, "<n>: matrix dimension (an nxn dense matrix is created)\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
    }

    int n;

    n = atoi(argv[1]);
    assert(n > 0);
    assert(n < 10000);

    /* ensure that n is a multiple of num_tasks */
    n = (n/num_tasks) * num_tasks;
    
    int n_p = (n/num_tasks);

    /* print new n to let user know n has been modified */
    if (rank == 0) {
        fprintf(stderr, "n: %d, n_p: %d\n", n, n_p);
        fprintf(stderr, "Requires %3.6lf MB of memory per task\n", ((3*4.0*n_p)*n/1e6));
    }

    float *A, *B, *C, *C_final;
    
    A = (float *) calloc(n_p * n, sizeof(float));
    assert(A != 0);

    B = (float *) calloc(n_p * n, sizeof(float));
    assert(B != 0);
    
    C = (float *) calloc(n_p * n, sizeof(float));
    assert(C != 0);

	C_final = (float *) calloc(n * n, sizeof(float));
	assert(C_final != 0);

    /* linearized matrices in row-major storage */
    /* A[i][j] would be A[i*n+j] */

    int i, j;

#ifdef _OPENMP
#pragma omp parallel for private(i,j)
#endif
	srand(time(NULL));
    for (i=0; i<n_p; i++) {
        for (j=0; j<n; j++) {
            A[i*n+j] = (float) rand() / (float) RAND_MAX;
            B[i*n+j] = (float) rand() / (float) RAND_MAX;
            C[i*n+j] = (float) rand() / (float) RAND_MAX;
        }
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    double elt = 0.0;
    if (rank == 0) 
        elt = timer();

#if USE_MPI
    /* Parallel matmul code goes here, see lecture slides for idea */

	int k;
	float *cur_rows = B;
	float *new_rows = (float *) malloc(n * n_p * sizeof(float));
	int send_rank = rank < num_tasks - 1 ? rank + 1 : 0;
	int recv_rank = rank > 0 ? rank - 1 : num_tasks - 1;
	float *C_tmp = (float *) calloc(n * n_p, sizeof(float));

	for (i = 0; i < n_p; i++) {
		cur_rows = B;
		memset(C_tmp, 0, n * n_p * sizeof(float));

		for (k = 0; k < n; k++) {
			for (j = 0; j < n; j++) {
				C[i * n + j] += A[i * n + k] * cur_rows[(k % n_p) * n + j];
			}

			if (k % n_p == n_p - 1) {
				MPI_Sendrecv(cur_rows, n * n_p, MPI_FLOAT, send_rank, 1,
						new_rows, n * n_p, MPI_FLOAT, recv_rank, 1, MPI_COMM_WORLD, &stat);

				if (cur_rows != B) {
					free(cur_rows);
				}

				cur_rows = new_rows;
				new_rows = (float *) malloc(n * n_p * sizeof(float));
			}
		}
	}

	free(cur_rows);
	free(new_rows);


#else
    int k;
    for (i=0; i<n_p; i++) {
        for (j=0; j<n; j++) {
            float c_ij = 0;
            for (k=0; k<n; k++) {
                c_ij += A[i*n+k]*B[k*n+j];
            }
            C[i*n+j] = c_ij;
        }
    }
#endif

    if (rank == 0) 
        elt = timer() - elt;

    /* Verify */
    int verify_failed = 0;
    for (i=0; i<n_p; i++) {
        for (j=0; j<n; j++) {
            if (C[i*n+j] != ((rank+1)*n)) {
                verify_failed = 1;
			}
        }
    }

    if (verify_failed) {
        fprintf(stderr, "ERROR: rank %d, verification failed, exiting!\n", rank);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        exit(2);
#endif
    }

    if (rank == 0) {
        fprintf(stderr, "Time taken: %3.3lf s.\n", elt);
        fprintf(stderr, "Performance: %3.3lf GFlop/s\n", (2.0*n*n)*n/(elt*1e9));
    }

    /* free memory */
    free(A); free(B); free(C); free(C_final);

    /* Shut down MPI */
#if USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
