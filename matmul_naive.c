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

// http://stackoverflow.com/a/13222410/6636682
inline float HorizontalSumAvx(const __m256 val) {
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

// http://stackoverflow.com/a/16743203/6636682
inline void transpose4x4_SSE(float *A, float *B, int n) {
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

inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int block_size) {
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

inline void transpose_block_SSE4x4_parallel(float *A, float *B, const int n, const int block_size) {
#pragma omp parallel for
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
    fprintf(stderr, "Requires %3.6lf MB memory\n", ((4*8.0*n)*n/1e6));

    float *A, *B, *B_T, *C;
    
	assert(posix_memalign((void **) &A, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &B, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &B_T, VECT_SIZE, n * n * sizeof(float)) == 0);
	assert(posix_memalign((void **) &C, VECT_SIZE, n * n * sizeof(float)) == 0);



    /* linearized matrices in row-major storage */
    /* A[i][j] would be A[i*n+j] */

    int i, j;

	srand(time(NULL));
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            A[i * n + j] = (float) rand() / (float) RAND_MAX;
            B[i * n + j] = (float) rand() / (float) RAND_MAX;
            C[i * n + j] = (float) rand() / (float) RAND_MAX;
        }
    }

    double elt = 0.0;
    elt = timer();

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
			// Transposing the second matrix allows us to use spatial locality
			// when calculating the sum of products
			transpose_block_SSE4x4(B, B_T, n, block_size);

			// https://blogs.msdn.microsoft.com/xiangfan/2009/04/28/optimize-your-code-matrix-multiplication/
			__m256 c;

			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					c = _mm256_setzero_ps();

					for (k = 0; k < n; k += VECT_NUM) {
						c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_load_ps(&A[i * n + k]), _mm256_load_ps(&B_T[j * n + k])));
					}

					C[i * n + j] += HorizontalSumAvx(c);
				}
			}

			break;
		case OPENMP:
			// Transposing the second matrix allows us to use spatial locality
			// when calculating the sum of products
			transpose_block_SSE4x4_parallel(B, B_T, n, block_size);

			// https://blogs.msdn.microsoft.com/xiangfan/2009/04/28/optimize-your-code-matrix-multiplication/
#pragma omp parallel for private(c)
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					c = _mm256_setzero_ps();

					for (k = 0; k < n; k += VECT_NUM) {
						c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_load_ps(&A[i * n + k]), _mm256_load_ps(&B_T[j * n + k])));
					}

					C[i * n + j] += HorizontalSumAvx(c);
				}
			}

			break;
	}


    elt = timer() - elt;
    fprintf(stderr, "Time taken: %3.3lf s.\n", elt);
    fprintf(stderr, "Performance: %3.3lf GFlop/s\n", (2.0*n*n*n)/(elt*1e9));

    /* free memory */
	free(A);
	free(B);
	free(B_T);
	free(C);

    return 0;
}
