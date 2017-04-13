#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <stdint.h>

#define	BASELINE			0
#define	BASELINE_OPTIMIZED	1
#define	OPENMP				2
#define	VECT_SIZE			32
#define	VECT_NUM			8

static double timer() {
    
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);

}

void print(float *tmp) {
	for (int i = 0; i < VECT_NUM / 2; i++) {
		printf("%f\t", tmp[i]);
	}
	printf("\n");
}
/*float HorizontalSumAvx(const __m256 val) {
	float *tmp = malloc(VECT_NUM * sizeof(float));
	const __m128 valupper = _mm256_extractf128_ps(val, 1);
	_mm_storeu_ps(tmp, valupper);
	print(tmp);
	const __m128 vallower = _mm256_extractf128_ps(val, 0);
	_mm_storeu_ps(tmp, vallower);
	print(tmp);
	_mm256_zeroupper();
	const __m128 valval = _mm_add_ps(valupper, vallower);
	_mm_storeu_ps(tmp, valval);
	print(tmp);
	__m128 valsum = _mm_add_ps(_mm_permute_ps(valval, 0x1B), valval);
	_mm_storeu_ps(tmp, valsum);
	print(tmp);
	__m128 res = _mm_add_ps(_mm_permute_ps(valsum, 0xB1), valval);
	_mm_storeu_ps(tmp, res);
	print(tmp);
	return _mm_cvtss_f32(res);
}*/

// http://stackoverflow.com/a/13222410/6636682
float HorizontalSumAvx(const __m256 val) {
	const __m128 hiQuad = _mm256_extractf128_ps(val, 1);
	const __m128 loQuad = _mm256_castps256_ps128(val);
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	const __m128 loDual = sumQuad;
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	const __m128 lo = sumDual;
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
}

// Adapted from http://stackoverflow.com/a/16743203/6636682
void transpose4x4_SSE(float *A, float *B, int n) {
	__m128 row1 = _mm_load_ps(&A[0 * n]);
	__m128 row2 = _mm_load_ps(&A[1 * n]);
	__m128 row3 = _mm_load_ps(&A[2 * n]);
	__m128 row4 = _mm_load_ps(&A[3 * n]);
	_MM_TRANSPOSE4_PS(row1, row2, row3, row4);
	_mm_store_ps(&B[0 * n], row1);
	_mm_store_ps(&B[1 * n], row2);
	_mm_store_ps(&B[2 * n], row3);
	_mm_store_ps(&B[3 * n], row4);
}

void transpose_block_SSE4x4(float *A, float *B, const int n, const int block_size) {
	for (int i = 0; i < n; i+= block_size) {
		for (int j = 0; j < n; j += block_size) {
			int max_i2 = i + block_size < n ? i + block_size : n;
			int max_j2 = j + block_size < n ? j + block_size : n;

			for (int i2 = i; i2 < max_i2; i2 += 4) {
				for (int j2 = j; j2 < max_j2; j2 += 4) {
					transpose4x4_SSE(&A[i2 * n + j2], &B[j2 * n + i2], n);
				}
			}
		}
	}
}

inline void transpose(int n, float *matrix) {
	int tmp;

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			tmp = matrix[i*n+j];
			matrix[i*n+j] = matrix[j*n+i];
			matrix[j*n+i] = tmp;
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

	// ensure that matrix dimensions are multiples of the number of elements
	// supported by vectorization
	n = (n + VECT_NUM - 1) / VECT_NUM * VECT_NUM;
	int block_size = n >= 16 ? 16 : 8;
	mat_mul_type = atoi(argv[2]);
	assert(mat_mul_type >= 0);
	assert(mat_mul_type < 3);

    fprintf(stderr, "n: %d\n", n);
    fprintf(stderr, "Requires %3.6lf MB memory\n", ((3*8.0*n)*n/1e6));

    float *A, *B, *B_T, *C;
	//float **C_ver;
    
	//A = (float **) malloc(n * sizeof(float *));
	//assert(A != NULL);
	//B = (float **) malloc(n * sizeof(float *));
	//assert(B != NULL);
	//C = (float **) malloc(n * sizeof(float *));
	//assert(C != NULL);
	assert(posix_memalign((void **) &A, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &B, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &B_T, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &C, VECT_SIZE, n * n * sizeof(float)) == 0);
	//C_ver = (float **) malloc(n * sizeof(float *));

	/*for (int i = 0; i < n; i++) {
		assert(posix_memalign((void **) &A[i], VECT_SIZE, n * sizeof(float)) == 0);
		assert((uintptr_t) A[i] % VECT_SIZE == 0);
		assert(posix_memalign((void **) &B[i], VECT_SIZE, n * sizeof(float)) == 0);
		assert((uintptr_t) B[i] % VECT_SIZE == 0);
		assert(posix_memalign((void **) &C[i], VECT_SIZE, n * sizeof(float)) == 0);
		assert((uintptr_t) C[i] % VECT_SIZE == 0);
		C_ver[i] = (float *) malloc(n * sizeof(float));
	}*/


    /* linearized matrices in row-major storage */
    /* A[i][j] would be A[i*n+j] */

    int i, j;

    /* static initalization, so that we can verify output */
    /* using very simple initialization right now */
    /* this isn't a good check for parallel debugging */
	srand(time(NULL));
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            A[i * n + j] = j;
            B[i * n + j] = j;
            C[i * n + j] = 0;
            //A[i * n + j] = (float) rand() / (float) RAND_MAX;
            //B[i * n + j] = (float) rand() / (float) RAND_MAX;
            //C[i * n + j] = (float) rand() / (float) RAND_MAX;
			//C_ver[i][j] = 0;
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
						c_ij += A[i*n+k]*B[k*n+j];
					}
					C[i*n+j] = c_ij;
				}
			}
			break;
		case BASELINE_OPTIMIZED:
			/*for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					float c_ij = 0;
					for (k=0; k<n; k++) {
						c_ij += A[i][k]*B[k][j];
					}
					C_ver[i][j] = c_ij;
				}
			}*/

			transpose_block_SSE4x4(B, B_T, n, block_size);
			//transpose(n, B);

			// Adapted from https://blogs.msdn.microsoft.com/xiangfan/2009/04/28/optimize-your-code-matrix-multiplication/
			__m256 c;
			//float *test = malloc(VECT_NUM * sizeof(float));
			//__m256 c_upper;
			//__m256 c_store;
			//__m256 c_load;
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					c = _mm256_setzero_ps();

					for (k = 0; k < n; k += VECT_NUM) {
						c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_load_ps(&A[i * n + k]), _mm256_load_ps(&B_T[j * n + k])));
					}

					C[i * n + j] += HorizontalSumAvx(c);
				}
			}
			//transpose(n, B);

			transpose_block_SSE4x4(B_T, B, n, block_size);
			break;
		case OPENMP:
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					float c_ij = 0;
#pragma omp parallel for reduction (+:c_ij)
					for (k=0; k<n; k++) {
						c_ij += A[i*n+k]*B[k*n+j];
					}
					C[i*n+j] = c_ij;
				}
			}
			break;
	}


    elt = timer() - elt;

    /* Verify */
    //int verify_failed = 0;
    /*for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (C[i][j] != C_ver[i][j])
                verify_failed = 1;
        }
    }*/

	/*for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d\t", (int)C[i*n+j]);
		}
		printf("\n");
	}*/
    /*if (verify_failed) {
        fprintf(stderr, "ERROR: verification failed, exiting!\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				/fprintf(stderr, "%d\t", (int) C[i][j]);
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
    }*/

    fprintf(stderr, "Time taken: %3.3lf s.\n", elt);
    fprintf(stderr, "Performance: %3.3lf GFlop/s\n", (2.0*n*n*n)/(elt*1e9));

    /* free memory */
	free(A);
	free(B);
	free(C);
	//free(C_ver);

    return 0;
}
