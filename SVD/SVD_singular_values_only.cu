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

	int M = 1000;
	int N = 1000;

	TimingGPU timerGPU;
	float     elapsedTime;

	// --- Setting the host matrix
	float *h_A = (float *)malloc(M * N * sizeof(float));
	for (unsigned int i = 0; i < M; i++){
		for (unsigned int j = 0; j < N; j++){
			h_A[j*M + i] = (i + j) * (i + j);
		}
	}

	// --- Setting the device matrix and moving the host matrix to the device
	float *d_A;         gpuErrchk(cudaMalloc(&d_A, M * N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));

	// --- host side SVD results space
	float *h_U = (float *)malloc(M * M * sizeof(float));
	float *h_V = (float *)malloc(N * N * sizeof(float));
	float *h_S = (float *)malloc(N *     sizeof(float));

	// --- device side SVD workspace and matrices
	int work_size = 0;

	int *devInfo;       gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));
	float *d_U;         gpuErrchk(cudaMalloc(&d_U, M * M * sizeof(float)));
	float *d_V;         gpuErrchk(cudaMalloc(&d_V, N * N * sizeof(float)));
	float *d_S;         gpuErrchk(cudaMalloc(&d_S, N *     sizeof(float)));

	cusolverStatus_t stat;

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolveSafeCall(cusolverDnCreate(&solver_handle));

	cusolveSafeCall(cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size));

	float *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));

	// --- CUDA SVD execution - Singular values only
	timerGPU.StartCounter();
	cusolveSafeCall(cusolverDnSgesvd(solver_handle, 'N', 'N', M, N, d_A, M, d_S, NULL, M, NULL, N, work, work_size, NULL, devInfo));
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
	//gpuErrchk(cudaMemcpy(h_S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < N; i++) std::cout << "d_S[" << i << "] = " << h_S[i] << std::endl;

	// --- CUDA SVD execution - Full SVD
	timerGPU.StartCounter();
	cusolveSafeCall(cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size, NULL, devInfo));
	elapsedTime = timerGPU.GetCounter();

	devInfo_h = 0;
	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h == 0)
		printf("SVD successfull for the full SVD calculation\n\n");
	else if (devInfo_h < 0)
		printf("SVD unsuccessfull for the full SVD calculation. Parameter %i is wrong\n", -devInfo_h);
	else
		printf("SVD unsuccessfull for the full SVD calculation. A number of %i superdiagonals of an intermediate bidiagonal form did not converge to zero\n", devInfo_h);

	printf("Calculation of the full SVD calculation: %f ms\n\n", elapsedTime);

	cusolveSafeCall(cusolverDnDestroy(solver_handle));

	return 0;

}
