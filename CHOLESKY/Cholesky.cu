#include "cuda_runtime.h"
#include "device_launch_paraMeters.h"

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main(){

	const int Nrows = 5;
	const int Ncols = 5;

	// --- Setting the host, Nrows x Ncols matrix
	double h_A[Nrows][Ncols] = { 
		{ 1.,    -1.,    -1.,    -1.,    -1.,},  
		{-1.,     2.,     0.,     0.,     0.,}, 
		{-1.,     0.,     3.,     1.,     1.,}, 
		{-1.,     0.,     1.,     4.,     2.,}, 
		{-1.,     0.,     1.,     2.,     5.,}
	};
	
	printf("Original matrix\n");
	for(int i = 0; i < Nrows; i++)
		for(int j = 0; j < Ncols; j++)
			printf("L[%i, %i] = %f\n", i, j, h_A[i][j]);

	// --- Setting the device matrix and moving the host matrix to the device
	double *d_A;			gpuErrchk(cudaMalloc(&d_A,		Nrows * Ncols * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;			gpuErrchk(cudaMalloc(&devInfo,	        sizeof(int)));
	
	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUDA CHOLESKY initialization
	cusolveSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, &work_size));

	// --- CUDA POTRF execution
	double *work;	gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));
	cusolveSafeCall(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, work, work_size, devInfo));
	int devInfo_h = 0;	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout	<< "Unsuccessful potrf execution\n\n";

	// --- At this point, the upper triangular part of A contains the elements of L. Showing this.
	printf("\nFactorized matrix\n");
	gpuErrchk(cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));
	for(int i = 0; i < Nrows; i++)
		for(int j = 0; j < Ncols; j++)
			if (i <= j) printf("L[%i, %i] = %f\n", i, j, h_A[i][j]);

	cusolverDnDestroy(solver_handle);

	return 0;

}
