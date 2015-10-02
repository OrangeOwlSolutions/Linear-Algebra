#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE
	cusparseHandle_t handle;	cusparseSafeCall(cusparseCreate(&handle));

	const int N     = 5;		// --- Size of the linear system

	// --- Lower diagonal, diagonal and upper diagonal of the system matrix
	double *h_ld = (double*)malloc(N * sizeof(double));
	double *h_d  = (double*)malloc(N * sizeof(double));
	double *h_ud = (double*)malloc(N * sizeof(double));
	
	h_ld[0]		= 0.;
	h_ud[N-1]	= 0.;
	for (int k = 0; k < N - 1; k++) {
		h_ld[k + 1] = -1.;
		h_ud[k]     = -1.;
	}

	for (int k = 0; k < N; k++) h_d[k] = 2.;

	double *d_ld;	gpuErrchk(cudaMalloc(&d_ld, N * sizeof(double)));
	double *d_d;	gpuErrchk(cudaMalloc(&d_d,  N * sizeof(double)));
	double *d_ud;	gpuErrchk(cudaMalloc(&d_ud, N * sizeof(double)));
	
	gpuErrchk(cudaMemcpy(d_ld, h_ld, N * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_d,  h_d,  N * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_ud, h_ud, N * sizeof(double), cudaMemcpyHostToDevice));
	
    // --- Allocating and defining dense host and device data vectors
	double *h_x	= (double *)malloc(N * sizeof(double)); 
	h_x[0] = 100.0;  h_x[1] = 200.0; h_x[2] = 400.0; h_x[3] = 500.0; h_x[4] = 300.0;

	double *d_x;		gpuErrchk(cudaMalloc(&d_x, N * sizeof(double)));   
    gpuErrchk(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
	
	// --- Allocating the host and device side result vector
	double *h_y	= (double *)malloc(N * sizeof(double)); 
	double *d_y;		gpuErrchk(cudaMalloc(&d_y, N * sizeof(double))); 

	cusparseSafeCall(cusparseDgtsv(handle, N, 1, d_ld, d_d, d_ud, d_x, N));

	cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
	for (int k=0; k<N; k++) printf("%f\n", h_x[k]);
}
