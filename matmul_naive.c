#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define	BASELINE			0
#define	BASELINE_OPTIMIZED	1
#define	OPENMP				2

static double timer() {
    
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);

}

// Adapted from http://stackoverflow.com/a/16743203/6636682
inline void transpose4x4_SSE(float **A) {
	__m128 row1 = _mm_load_ps(A[0]);
	__m128 row2 = _mm_load_ps(A[1]);
	__m128 row3 = _mm_load_ps(A[2]);
	__m128 row4 = _mm_load_ps(A[3]);
	_MM_TRANSPOSE4_PS(row1, row2, row3, row4);
	_mm_store_ps(A[0], row1);
	_mm_store_ps(A[1], row2);
	_mm_store_ps(A[2], row3);
	_mm_store_ps(A[3], row4);
}

inline void transpose_block_SSE4x4(float **A, const int n, const int block_size) {
	for (int i = 0; i < n; i+= block_size) {
		for (int j = 0; j < n; j += block_size) {
			int max_i2 = i + block_size < n ? i + block_size : n;
			int max_j2 = j + block_size < n ? j + block_size : n;

			for (int i2 = i; i2 < max_i2; i2 += 4) {
				for (int j2 = j; j2 < max_j2; j2 += 4) {
					transpose4x4_SSE(&(&A[i2])[j2]);
				}
			}
		}
	}
}

inline void transpose(int n, float **matrix) {
	int tmp;

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			tmp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = tmp;
		}
	}
}

int main(int argc, char **argv) {

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "%s <n> <t>\n", argv[0]);
        fprintf(stderr, "<n>: matrix dimension (nxn dense matrices are created)\n");
		fprintf(stderr, "<t>: the matrix matrix algorithm used\n");
		fprintf(stderr, "\t0: (default) naive serial algorithm without loop unrolling and vectorization\n");
		fprintf(stderr, "\t1: serial algorithm with loop unrolling and vectorization\n");
		fprintf(stderr, "\t2: parallelized algorithm with OpenMP\n");
        exit(1);
    }

    int n;
	int mat_mul_type = 0;

    n = atoi(argv[1]);
    assert(n > 0);
    assert(n < 10000);

	mat_mul_type = atoi(argv[2]);
	assert(mat_mul_type >= 0);
	assert(mat_mul_type < 3);

    fprintf(stderr, "n: %d\n", n);
    fprintf(stderr, "Requires %3.6lf MB memory\n", ((3*8.0*n)*n/1e6));

    float **A, **B, **C;
	float **C_ver;
    
	//A = (float **) malloc(n * sizeof(float *));
	//assert(A != NULL);
	//B = (float **) malloc(n * sizeof(float *));
	//assert(B != NULL);
	//C = (float **) malloc(n * sizeof(float *));
	//assert(C != NULL);
	assert(posix_memalign((void **) &A, 32, n * sizeof(float *)) == 0);
	assert(posix_memalign((void **) &B, 32, n * sizeof(float *)) == 0);
	assert(posix_memalign((void **) &C, 32, n * sizeof(float *)) == 0);
	C_ver = (float **) malloc(n * sizeof(float *));

	for (int i = 0; i < n; i++) {
		assert(posix_memalign((void **) &A[i], 32, n * sizeof(float)) == 0);
		assert(posix_memalign((void **) &B[i], 32, n * sizeof(float)) == 0);
		assert(posix_memalign((void **) &C[i], 32, n * sizeof(float)) == 0);
		C_ver[i] = (float *) malloc(n * sizeof(float));
	}


    /* linearized matrices in row-major storage */
    /* A[i][j] would be A[i*n+j] */

    int i, j;

    /* static initalization, so that we can verify output */
    /* using very simple initialization right now */
    /* this isn't a good check for parallel debugging */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            A[i][j] = j;
            B[i][j] = j;
            C[i][j] = 0;
			C_ver[i][j] = 0;
        }
    }

    double elt = 0.0;
    elt = timer();

	// TODO add option to parallelize
    int k = 0;
	switch(mat_mul_type) {
		case BASELINE:
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					float c_ij = 0;
					for (k=0; k<n; k++) {
						c_ij += A[i][k]*B[k][j];
					}
					C[i][j] = c_ij;
				}
			}
			break;
		case BASELINE_OPTIMIZED:
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					float c_ij = 0;
					for (k=0; k<n; k++) {
						c_ij += A[i][k]*B[k][j];
					}
					C_ver[i][j] = c_ij;
				}
			}

			//transpose_block_SSE4x4(B, n, 16);
			transpose(n, B);

			// Adapted from https://blogs.msdn.microsoft.com/xiangfan/2009/04/28/optimize-your-code-matrix-multiplication/
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					__m256 c = _mm256_setzero_ps();
					float c_ij = 0;
					k = 0;
					for (; k<n; k+=8) {
						//printf("there\n");
						c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_load_ps(&A[i][k]), _mm256_load_ps(&B[j][k])));
					}

					//if (j + 7 < n) {
						//printf("here\n");
						c = _mm256_hadd_ps(c, c);
						c = _mm256_hadd_ps(c, c);
						c = _mm256_hadd_ps(c, c);
						_mm256_storeu_ps(&C[i][j], _mm256_add_ps(_mm256_load_ps(&C[i][j]), c));
						_mm256_storeu_ps(&C[i][j], c);
					//}

					/*for (; k < n; k++) {
						c_ij += A[i][k]*B[j][k];
						//printf("%f * %f\t", A[i][k], B[j][k]);
					}*/
					//printf("\n");

					C[i][j] += c_ij;
				}
			}
			transpose(n, B);

			//transpose_block_SSE4x4(B, n, 16);
			break;
		case OPENMP:
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					float c_ij = 0;
#pragma omp parallel for reduction (+:c_ij)
					for (k=0; k<n; k++) {
						c_ij += A[i][k]*B[k][j];
					}
					C[i][j] = c_ij;
				}
			}
			break;
	}


    elt = timer() - elt;

    /* Verify */
    int verify_failed = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (C[i][j] != C_ver[i][j])
                verify_failed = 1;
        }
    }

    if (verify_failed) {
        fprintf(stderr, "ERROR: verification failed, exiting!\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				fprintf(stderr, "%d\t", (int) C[i][j]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				fprintf(stderr, "%d\t", (int) C_ver[i][j]);
			}
			fprintf(stderr, "\n");
		}
        exit(2);
    }

    fprintf(stderr, "Time taken: %3.3lf s.\n", elt);
    fprintf(stderr, "Performance: %3.3lf GFlop/s\n", (2.0*n*n*n)/(elt*1e9));

    /* free memory */
	for (i = 0; i < n; i++) {
		free(A[i]);
		free(B[i]);
		free(C[i]);
		free(C_ver[i]);
	}

	free(A);
	free(B);
	free(C);
	free(C_ver);

    return 0;
}
