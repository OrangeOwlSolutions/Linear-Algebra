#include <stdio.h>

#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include "Utilities.cuh"

int main() {

	const unsigned int N = 3; 

	const unsigned int Nmatrices = 1;
	
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));

	// --- Matrices to be inverted (only one in this example)
	float *h_A = new float[N*N*Nmatrices];
	
	h_A[0] = 4.f;  
	h_A[1] = 3.f;
	h_A[2] = 8.f;
	h_A[3] = 9.f;
	h_A[4] = 5.f; 
	h_A[5] = 1.f; 
	h_A[6] = 2.f; 
	h_A[7] = 7.f;
	h_A[8] = 6.f;

	// --- Allocate device matrices 
	float *d_A;	gpuErrchk(cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float)));

	// --- Move the matrix to be inverted from host to device
	gpuErrchk(cudaMemcpy(d_A,h_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice));

	// --- Creating the array of pointers needed as input to the batched getrf
	float **h_inout_pointers = (float **)malloc(Nmatrices*sizeof(float *));
	for (int i=0; i<Nmatrices; i++) h_inout_pointers[i]=(float *)((char*)d_A+i*((size_t)N*N)*sizeof(float));
 
	float **d_inout_pointers;
	gpuErrchk(cudaMalloc((void**)&d_inout_pointers, Nmatrices*sizeof(float *)));
	gpuErrchk(cudaMemcpy(d_inout_pointers,h_inout_pointers,Nmatrices*sizeof(float *),cudaMemcpyHostToDevice));
	free(h_inout_pointers);

	int *d_PivotArray; gpuErrchk(cudaMalloc((void**)&d_PivotArray, N*Nmatrices*sizeof(int)));
	int *d_InfoArray;  gpuErrchk(cudaMalloc((void**)&d_InfoArray,  Nmatrices*sizeof(int)));
	
	int *h_PivotArray = (int *)malloc(N*Nmatrices*sizeof(int));
	int *h_InfoArray  = (int *)malloc(  Nmatrices*sizeof(int));
	
	cublasSafeCall(cublasSgetrfBatched(handle, N, d_inout_pointers, N, d_PivotArray, d_InfoArray, Nmatrices));
	//cublasSafeCall(cublasSgetrfBatched(handle, N, d_inout_pointers, N, NULL, d_InfoArray, Nmatrices));
	
    gpuErrchk(cudaMemcpy(h_InfoArray,d_InfoArray,Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nmatrices; i++)
		if (h_InfoArray[i]  != 0) {
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	gpuErrchk(cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_PivotArray,d_PivotArray,N*Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

	for (int i=0; i<N*N; i++) printf("A[%i]=%f\n", i, h_A[i]);

	printf("\n\n");
	for (int i=0; i<N; i++) printf("P[%i]=%i\n", i, h_PivotArray[i]);
	
	return 0;
}
