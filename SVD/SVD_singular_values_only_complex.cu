#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

/********/
/* MAIN */
/********/
int main(){

	int M = 64;
	int N = 64;

	TimingGPU timerGPU;
	float     elapsedTime;

	// --- Setting the host matrix
	float2 *h_A = (float2 *)malloc(M * N * sizeof(float2));
	for (unsigned int i = 0; i < M; i++){
		for (unsigned int j = 0; j < N; j++){
			h_A[j*M + i].x = (i + j) * (i + j);
			h_A[j*M + i].y = 2 * sqrt((i + j) * (i + j));
		}
	}

	// --- Setting the device matrix and moving the host matrix to the device
	float2 *d_A;         gpuErrchk(cudaMalloc(&d_A, M * N * sizeof(float2)));
	gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(float2), cudaMemcpyHostToDevice));

	// --- host side SVD results space
	float2 *h_U = (float2 *)malloc(M * M * sizeof(float2));
	float2 *h_V = (float2 *)malloc(N * N * sizeof(float2));
	float  *h_S = (float  *)malloc(N *     sizeof(float));

	// --- device side SVD workspace and matrices
	int work_size = 0;

	int *devInfo;       gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
	float2 *d_U;        gpuErrchk(cudaMalloc(&d_U, M * M * sizeof(float2)));
	float2 *d_V;        gpuErrchk(cudaMalloc(&d_V, N * N * sizeof(float2)));
	float *d_S;         gpuErrchk(cudaMalloc(&d_S, N *     sizeof(float)));

	cusolverStatus_t stat;

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolveSafeCall(cusolverDnCreate(&solver_handle));

	cusolveSafeCall(cusolverDnCgesvd_bufferSize(solver_handle, M, N, &work_size));

	float2 *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float2)));

	// --- CUDA SVD execution - Singular values only
	timerGPU.StartCounter();
	cusolveSafeCall(cusolverDnCgesvd(solver_handle, 'N', 'N', M, N, (cuComplex *)d_A, M, d_S, NULL, M, NULL, N, (cuComplex *)work, work_size, NULL, devInfo));
	elapsedTime = timerGPU.GetCounter();

	int devInfo_h = 0;
	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h == 0)
		printf("SVD successfull for the singular values calculation only\n\n");
	else if (devInfo_h < 0)
		printf("SVD unsuccessfull for the singular values calculation only. Parameter %i is wrong\n", -devInfo_h);
	else
		printf("SVD unsuccessfull for the singular values calculation only. A number of %i superdiagonals of an intermediate bidiagonal form did not converge to zero\n", devInfo_h);

	printf("Calculation of the singular values only: %f ms\n\n", elapsedTime);

	// --- Moving the results from device to host
	gpuErrchk(cudaMemcpy(h_S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i++) std::cout << "d_S[" << i << "] = " << h_S[i] << std::endl;

	cusolveSafeCall(cusolverDnDestroy(solver_handle));

	return 0;

}
