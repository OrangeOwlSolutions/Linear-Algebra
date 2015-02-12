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

#define BLOCK_SIZE 32

/***************/
/* COPY KERNEL */
/***************/
__global__ void copy_kernel(const double * __restrict d_in1, double * __restrict d_out1, const double * __restrict d_in2, double * __restrict d_out2, const int M, const int N) {

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N)) {
		d_out1[j * N + i] = d_in1[j * M + i];
		d_out2[j * N + i] = d_in2[j * M + i];
	}
}

/********/
/* MAIN */
/********/
int main(){

	// --- ASSUMPTION Nrows >= Ncols
	
	const int Nrows = 7;
	const int Ncols = 5;

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;			gpuErrchk(cudaMalloc(&devInfo,	        sizeof(int)));
	
	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUBLAS initialization
	cublasHandle_t cublas_handle;
	cublasSafeCall(cublasCreate(&cublas_handle));

	// --- Setting the host, Nrows x Ncols matrix
	double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
	for(int j = 0; j < Nrows; j++)
		for(int i = 0; i < Ncols; i++)
			h_A[j + i*Nrows] = (i + j*j) * sqrt((double)(i + j));

	// --- Setting the device matrix and moving the host matrix to the device
	double *d_A;			gpuErrchk(cudaMalloc(&d_A,		Nrows * Ncols * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

	// --- CUDA QR initialization
	double *d_TAU;		gpuErrchk(cudaMalloc((void**)&d_TAU, min(Nrows, Ncols) * sizeof(double)));
	cusolveSafeCall(cusolverDnDgeqrf_bufferSize(solver_handle, Nrows, Ncols, d_A, Nrows, &work_size));
	double *work;	gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

	// --- CUDA GERF execution
	cusolveSafeCall(cusolverDnDgeqrf(solver_handle, Nrows, Ncols, d_A, Nrows, d_TAU, work, work_size, devInfo));
	int devInfo_h = 0;	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout	<< "Unsuccessful gerf execution\n\n";

	// --- At this point, the upper triangular part of A contains the elements of R. Showing this.
	gpuErrchk(cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));
	for(int j = 0; j < Nrows; j++)
		for(int i = 0; i < Ncols; i++)
			if (i >= j) printf("R[%i, %i] = %f\n", j, i, h_A[j + i*Nrows]);

	// --- Initializing the output Q matrix (Of course, this step could be done by a kernel function directly on the device)
	double *h_Q = (double *)malloc(Nrows * Nrows * sizeof(double));
	for(int j = 0; j < Nrows; j++)
		for(int i = 0; i < Nrows; i++)
			if (j == i) h_Q[j + i*Nrows] = 1.;
			else		h_Q[j + i*Nrows] = 0.;

	double *d_Q;			gpuErrchk(cudaMalloc(&d_Q,		Nrows * Nrows * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_Q, h_Q, Nrows * Nrows * sizeof(double), cudaMemcpyHostToDevice));

	// --- CUDA QR execution
	cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_Q, Nrows, work, work_size, devInfo));
	
	// --- At this point, d_Q contains the elements of Q. Showing this.
	gpuErrchk(cudaMemcpy(h_Q, d_Q, Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\n");
	for(int j = 0; j < Nrows; j++)
		for(int i = 0; i < Nrows; i++)
			printf("Q[%i, %i] = %f\n", j, i, h_Q[j + i*Nrows]);
	
	// --- Initializing the data matrix C (Of course, this step could be done by a kernel function directly on the device).
	// --- Notice that, in this case, only the first column of C contains actual data, the others being empty (zeroed). However, cuBLAS trsm
	//     has the capability of solving triangular linear systems with multiple right hand sides.
	double *h_C = (double *)calloc(Nrows * Nrows, sizeof(double));
	for(int j = 0; j < Nrows; j++)
		h_C[j] = 1.;

	double *d_C;			gpuErrchk(cudaMalloc(&d_C,		Nrows * Nrows * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_C, h_C, Nrows * Nrows * sizeof(double), cudaMemcpyHostToDevice));

	// --- CUDA QR execution
	cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_C, Nrows, work, work_size, devInfo));
	
	// --- At this point, d_C contains the elements of Q^T * C, where C is the data vector. Showing this.
	// --- According to the above, only the first column of d_C makes sense.
	gpuErrchk(cudaMemcpy(h_C, d_C, Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\n");
	for(int j = 0; j < Nrows; j++)
		for(int i = 0; i < Nrows; i++)
			printf("C[%i, %i] = %f\n", j, i, h_C[j + i*Nrows]);

	// --- Reducing the linear system size
	double *d_R; gpuErrchk(cudaMalloc(&d_R, Ncols * Ncols * sizeof(double)));
	double *h_B = (double *)malloc(Ncols * Ncols * sizeof(double));
	double *d_B; gpuErrchk(cudaMalloc(&d_B, Ncols * Ncols * sizeof(double)));
	dim3 Grid(iDivUp(Ncols, BLOCK_SIZE), iDivUp(Ncols, BLOCK_SIZE));
	dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
	copy_kernel<<<Grid, Block>>>(d_A, d_R, d_C, d_B, Nrows, Ncols);

	// --- Solving an upper triangular linear system
	const double alpha = 1.;
	cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Ncols, Ncols,
		                       &alpha, d_R, Ncols, d_B, Ncols));

	gpuErrchk(cudaMemcpy(h_B, d_B, Ncols * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

	printf("\n\n");
	for (int i=0; i<Ncols; i++) printf("B[%i] = %f\n", i, h_B[i]);

	cusolverDnDestroy(solver_handle);

	return 0;

}

