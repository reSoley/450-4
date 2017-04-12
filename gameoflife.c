#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define USE_MPI 1

#if USE_MPI
#include <mpi.h>
#endif

static double timer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

void display_grid(int *grid, int m_p, int m) {
	int i, j, cur;

	for (i = 0; i < m_p; i++) {
		for (j = 0; j < m; j++) {
			cur = grid[(i * m) + j];

			if (cur == 0) {
				printf(".");
			} else {
				printf("O");
			}
		}
		printf("\n");
	}
	printf("\n");
	usleep(70000);
}

int main(int argc, char **argv) {

    int rank, num_tasks;

    /* Initialize MPI */
#if USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("Hello world from rank %3d of %3d\n", rank, num_tasks);
#else
    rank = 0;
    num_tasks = 1;
#endif

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "%s <m> <k> -p\n", argv[0]);
            fprintf(stderr, "Program for parallel Game of Life\n");
            fprintf(stderr, "with 1D grid partitioning\n");
            fprintf(stderr, "<m>: grid dimension (an mxm grid is created)\n");
            fprintf(stderr, "<k>: number of time steps\n");
	    fprintf(stderr, "-p : display the game grid\n");
	    fprintf(stderr, "     **Alters timing results**\n");
            fprintf(stderr, "(initial pattern specified inside code)\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
    }

    int m, k;

    m = atoi(argv[1]);
    assert(m > 2);
    assert(m <= 10000);

    k = atoi(argv[2]);
    assert(k > 0);
    assert(k <= 1000);

    int is_print = 0;
    if (argc == 4) {
	    if (strcmp(argv[3], "-p") == 0) {
		is_print = 1;
	    }
    }

    /* ensure that m is a multiple of num_tasks */
    m = (m/num_tasks) * num_tasks;
    
    int m_p = (m/num_tasks);

    /* print new m to let user know n has been modified */
    if (rank == 0) {
        fprintf(stderr, "Using m: %d, m_p: %d, k: %d\n", m, m_p, k);
        fprintf(stderr, "Requires %3.6lf MB of memory per task\n", 
                ((2*4.0*m_p)*m/1e6));
    }

    /* Linearizing 2D grids to 1D using row-major ordering */
    /* grid[i][j] would be grid[i*n+j] */
    int *grid_current;
    int *grid_next;
    
    grid_current = (int *) malloc(m_p * m * sizeof(int));
    assert(grid_current != 0);

    grid_next = (int *) malloc(m_p * m * sizeof(int));
    assert(grid_next != 0);

    int i, j, t;

#ifdef _OPENMP
#pragma omp parallel for private(i,j)
#endif
    for (i=0; i<m_p; i++) {
        for (j=0; j<m; j++) {
            grid_current[i*m+j] = 0;
            grid_next[i*m+j] = 0;
        }
    }

    /* initializing random cells */
    srand(time(NULL)+rank);

    for (i=0; i<m_p; i++) {
        for (j=1; j<m-1; j++) {
	    /* 1/n chance of cell to start alive */
	    int n = 5;
	    int r = rand() % n;

	    if (r == 0) {
            	grid_current[i*m+j] = 1;
            	grid_next[i*m+j] = 1;
	    }
        }
    }

    /* initializing some cells in the middle */
    /*
    assert((m*m_p/2 + m/2 + 3) < m_p*m);
    grid_current[m*m_p/2 + m/2 + 0] = 1;
    grid_current[m*m_p/2 + m/2 + 1] = 1;
    grid_current[m*m_p/2 + m/2 + 2] = 1;
    grid_current[m*m_p/2 + m/2 + 3] = 1;
    */

    /* initializing blinkers on each border */
    /*
    assert(m > 5);
    int center_w = m/2;
    if (rank % 2 == 0) {
    	grid_current[(m_p-1)*m+center_w-1] = 1;
    	grid_current[(m_p-1)*m+center_w] = 1;
    	grid_current[(m_p-1)*m+center_w+1] = 1;
    }
    */

    /* initializing glider in each region */
    /*
    assert(m > 3);
    assert(m_p > 3);
    grid_current[m+2] = 1;
    grid_current[2*m+3] = 1;
    grid_current[3*m+1] = 1;
    grid_current[3*m+2] = 1;
    grid_current[3*m+3] = 1;
    */

    /* initializing glider gun in top region */
    /*
    assert(m > 40);
    assert(m_p > 10);
    if (rank == 0) {
        int center = m/2;
        grid_current[2*m+center+6] = 1;
        grid_current[3*m+center+4] = 1;
        grid_current[3*m+center+6] = 1;
        grid_current[4*m+center-6] = 1;
        grid_current[4*m+center-5] = 1;
        grid_current[4*m+center+2] = 1;
        grid_current[4*m+center+3] = 1;
        grid_current[4*m+center+16] = 1;
        grid_current[4*m+center+17] = 1;
        grid_current[5*m+center-7] = 1;
        grid_current[5*m+center-3] = 1;
        grid_current[5*m+center+2] = 1;
        grid_current[5*m+center+3] = 1;
        grid_current[5*m+center+16] = 1;
        grid_current[5*m+center+17] = 1;
        grid_current[6*m+center-17] = 1;
        grid_current[6*m+center-18] = 1;
        grid_current[6*m+center-8] = 1;
        grid_current[6*m+center-2] = 1;
        grid_current[6*m+center+2] = 1;
        grid_current[6*m+center+3] = 1;
        grid_current[7*m+center-17] = 1;
        grid_current[7*m+center-18] = 1;
        grid_current[7*m+center-8] = 1;
        grid_current[7*m+center-4] = 1;
        grid_current[7*m+center-2] = 1;
        grid_current[7*m+center-1] = 1;
        grid_current[7*m+center+4] = 1;
        grid_current[7*m+center+6] = 1;
        grid_current[8*m+center-8] = 1;
        grid_current[8*m+center-2] = 1;
        grid_current[8*m+center+6] = 1;
        grid_current[9*m+center-7] = 1;
        grid_current[9*m+center-3] = 1;
        grid_current[10*m+center-6] = 1;
        grid_current[10*m+center-5] = 1;
    }
    */

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    double elt = 0.0;
    if (rank == 0) 
        elt = timer();

#if USE_MPI
	/* Parallel code goes here */
	/* grid_current and grid_next must be updated correctly */

	/* Allocate rows for border rows that must be shared */
	MPI_Status status_top, status_bottom;
	int *top_row = (int *) malloc(m * sizeof(int));
	int *bottom_rom = (int *) malloc(m * sizeof(int));
	int send_up, send_down, recv_up, recv_down, num_alive, prev_state;

	/* Set send and receive processes (circular) */
	if (rank == 0) {
		send_up = num_tasks - 1;
		recv_down = num_tasks - 1;
	} else {
		send_up = rank - 1;
		recv_down = rank - 1;
	}

	if (rank == num_tasks - 1) {
		recv_up = 0;
		send_down = 0;
	} else {
		recv_up = rank + 1;
		send_down = rank + 1;
	}

	/* left and right edges fixed at 0 */
	/* top and bottom wrap around */
	double t_comm = 0.0;
	for (t=0; t<k; t++ ) {
		/* Display current state of game (won't necessarily print rows in order) */

		if (is_print) {
			/* WARNING: drastically alters timing */
			int *grid_master = (int *)malloc(m * m * sizeof(int));
			MPI_Allgather(grid_current, m*m_p, MPI_INT,
					grid_master, m*m_p, MPI_INT, MPI_COMM_WORLD);

			if (rank == 0) {
				display_grid(grid_master, m, m);
			}
		}

		//Measure time spent in communication to determine fraction of overall
		double t_k;
		if (rank == 0) {
			t_k = timer();
		}

		/* Sends the top row of each rank's section to the preceding rank */
    		MPI_Sendrecv(grid_current, m, MPI_INT, send_up, 123, 
			    	bottom_rom, m, MPI_INT, recv_up, 123, MPI_COMM_WORLD, &status_top);

		/* Sends the bottom row of each rank's section to the following rank */
    		MPI_Sendrecv(grid_current+((m_p-1)*m), m, MPI_INT, send_down, 456, 
			    	top_row, m, MPI_INT, recv_down, 456, MPI_COMM_WORLD, &status_bottom);

		if (rank == 0) { 
			t_k = timer() - t_k;
			t_comm = t_comm + t_k;
		}

		for (i = 0; i < m_p; i++) {
			for (j = 1; j < m-1; j++) {
				prev_state = grid_current[i*m+j];
				num_alive = 0;

				if (i == 0) {			/* Use shared row to set top row */
					num_alive +=
                                		top_row[j-1] + 
                                		top_row[j] + 
                                		top_row[j+1];
				} else {
					num_alive +=
                                		grid_current[(i-1)*m+j-1] + 
                                		grid_current[(i-1)*m+j] + 
                                		grid_current[(i-1)*m+j+1];
				}

				num_alive +=
                                	grid_current[i*m+j-1] + 
                                	grid_current[i*m+j+1];

				if (i == m_p - 1) {		/* Use shared row to set bottom row */
					num_alive +=
						bottom_rom[j-1] + 
						bottom_rom[j] + 
						bottom_rom[j+1];
				} else {
					num_alive +=
                                		grid_current[(i+1)*m+j-1] + 
                                		grid_current[(i+1)*m+j] + 
                                		grid_current[(i+1)*m+j+1];
				}

                		grid_next[i*m+j] = prev_state * ((num_alive == 2) + (num_alive == 3)) +
					(1 - prev_state) * (num_alive == 3);
			}
		}
 
		/* Swap current and next */
		int *grid_tmp  = grid_next;
		grid_next = grid_current;
		grid_current = grid_tmp;
	}

#else
    /* serial code */
    /* considering only internal cells */
    for (t=0; t<k; t++) {
	if (is_print) {
		/* WARNING: drastically alters timing results */
		display_grid(grid_current, m, m);
	}

        for (i=1; i<m-1; i++) {
            for (j=1; j<m-1; j++) {
                /* avoiding conditionals inside inner loop */
                int prev_state = grid_current[i*m+j];
                int num_alive  = 
                                grid_current[(i  )*m+j-1] + 
                                grid_current[(i  )*m+j+1] + 
                                grid_current[(i-1)*m+j-1] + 
                                grid_current[(i-1)*m+j  ] + 
                                grid_current[(i-1)*m+j+1] + 
                                grid_current[(i+1)*m+j-1] + 
                                grid_current[(i+1)*m+j  ] + 
                                grid_current[(i+1)*m+j+1];

                grid_next[i*m+j] = prev_state * ((num_alive == 2) + (num_alive == 3)) +
			(1 - prev_state) * (num_alive == 3);
            }
        }

        /* swap current and next */
        int *grid_tmp  = grid_next;
        grid_next = grid_current;
        grid_current = grid_tmp;
    }
#endif

    if (rank == 0) 
        elt = timer() - elt;

    /* Verify */
    /* Used '-p' flag for visual verification */
    /* More versatile than hard coding a few concrete solutions */

    /*
    int verify_failed = 0;
    for (i=0; i<m_p; i++) {
        for (j=0; j<m; j++) {
            // Add verification code here
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
    */

    if (rank == 0) {
        fprintf(stderr, "Time taken: %3.3lf s.\n", elt);
#if USE_MPI
	fprintf(stderr, "Time spent in communication: %3.3lf s.\n", t_comm);
	fprintf(stderr, "Fraction of time spent in communication: %3.3lf\n", t_comm / elt);
#endif
        fprintf(stderr, "Performance: %3.3lf billion cell updates/s\n", 
                (1.0*m*m)*k/(elt*1e9));
    }

    /* free memory */
    free(grid_current); free(grid_next);

    /* Shut down MPI */
#if USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
