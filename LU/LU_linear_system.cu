#include <stdio.h>

#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include "Utilities.cuh"

float *createPermutationMatrix(int *h_PivotArray, int N)
{
    int temp;

    // --- Create permutation matrix
	float *P = (float *)malloc(N * N * sizeof(float));
	for (int i=0; i<N; i++) {
		P[i] = 0.0f;
		for (int j=0; j<N; j++) 
			if (i == j) P[i * N + j] = 1.0f;
	}

	for (int j=0; j<N; j++) 
		for (int i=0; i<N-1; i++) {
			temp						= P[i + j * N];
			P[i + j * N]				= P[(h_PivotArray[i] - 1) + j * N];
			P[(h_PivotArray[i] - 1) + j * N]	= temp;
		}

	return P;
}

/********/
/* MAIN */
/********/
int main() {

	const unsigned int N = 3; 

	const unsigned int Nmatrices = 1;
	
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));

	/***********************/
	/* SETTING THE PROBLEM */
	/***********************/

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

	// --- Known term (only one in this example)
	float *h_B = new float[N];

	h_B[0] = 1.f;
	h_B[1] = 0.5f;
	h_B[2] = 3.;
	
	// --- Result (only one in this example)
	float *h_X = new float[N];

	// --- Allocate device space for the input matrices 
	float *d_A;	gpuErrchk(cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float)));
	float *d_B;	gpuErrchk(cudaMalloc((void**)&d_B, N*            sizeof(float)));
	float *d_X;	gpuErrchk(cudaMalloc((void**)&d_X, N*            sizeof(float)));

	// --- Move the relevant matrices from host to device
	gpuErrchk(cudaMemcpy(d_A,h_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B,h_B,N*            sizeof(float),cudaMemcpyHostToDevice));

	/********************/
	/* LU DECOMPOSITION */
	/********************/
	
	// --- Creating the array of pointers needed as input/output to the batched getrf
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

	// --- The output factored matrix in column-major format
	for (int i=0; i<N*N; i++) printf("A[%i]=%f\n", i, h_A[i]);

	printf("\n\n");
	// --- The pivot array
	for (int i=0; i<N; i++) printf("IPIV[%i]=%i\n", i, h_PivotArray[i]);
	
	/*******************************************/
	/* APPROACH NR.1: THROUGH THE INVERSE OF A */
	/*******************************************/

	// --- Allocate host space for the inverted matrices 
	float *h_C = new float[N*N*Nmatrices];

	// --- Allocate device space for the inverted matrices 
	float *d_C;	gpuErrchk(cudaMalloc((void**)&d_C, N*N*Nmatrices*sizeof(float)));

	// --- Creating the array of pointers needed as output to the batched getri
	float **h_out_pointers = (float **)malloc(Nmatrices*sizeof(float *));
	for (int i=0; i<Nmatrices; i++) h_out_pointers[i]=(float *)((char*)d_C+i*((size_t)N*N)*sizeof(float));
 
	float **d_out_pointers;
	gpuErrchk(cudaMalloc((void**)&d_out_pointers, Nmatrices*sizeof(float *)));
	gpuErrchk(cudaMemcpy(d_out_pointers,h_out_pointers,Nmatrices*sizeof(float *),cudaMemcpyHostToDevice));
	free(h_out_pointers);

	cublasSafeCall(cublasSgetriBatched(handle, N, (const float **)d_inout_pointers, N, d_PivotArray, d_out_pointers, N, d_InfoArray, Nmatrices));

    gpuErrchk(cudaMemcpy(h_InfoArray,d_InfoArray,Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nmatrices; i++)
		if (h_InfoArray[i]  != 0) {
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	gpuErrchk(cudaMemcpy(h_C,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost));
	
	// --- The output inverted matrix in column-major format
	printf("\n\n");
	for (int i=0; i<N*N; i++) printf("C[%i]=%f\n", i, h_C[i]);
	
	float alpha1  = 1.f;
	float beta1   = 0.f;

	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha1, d_C, N, d_B, 1, &beta1, d_X, 1));

	gpuErrchk(cudaMemcpy(h_X,d_X,N*sizeof(float),cudaMemcpyDeviceToHost));
	
	// --- The output inverted matrix in column-major format
	printf("\n\n");
	for (int i=0; i<N; i++) printf("X[%i]=%f\n", i, h_X[i]);

	/*****************************************************************************/
	/* APPROACH NR.2: THROUGH THE INVERSE OF UPPER AND LOWER TRIANGULAR MATRICES */
	/*****************************************************************************/

	float *P = createPermutationMatrix(h_PivotArray, N);

	float *d_P; gpuErrchk(cudaMalloc((void**)&d_P, N * N * sizeof(float)));

	printf("\n\n");
	// --- The permutation matrix
	for (int i=0; i<N; i++) 
		for (int j=0; j<N; j++)
			printf("P[%i, %i]=%f\n", i, j, P[j * N + i]);

	gpuErrchk(cudaMemcpy(d_P, P, N * N * sizeof(float), cudaMemcpyHostToDevice));
	
	// --- Now P*A=L*U
	//     Linear system A*x=y => P.'*L*U*x=y => L*U*x=P*y

	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha1, d_P, N, d_B, 1, &beta1, d_B, 1));

	gpuErrchk(cudaMemcpy(h_B,d_B,N*sizeof(float),cudaMemcpyDeviceToHost));
	
	// --- The result of P*y
	printf("\n\n");
	for (int i=0; i<N; i++) printf("(P*y)[%i]=%f\n", i, h_B[i]);

	// --- 1st phase - solve Ly = b 
	const float alpha  = 1.f;

	// --- Function solves the triangulatr linear system with multiple right hand sides, function overrides b as a result 

	// --- Lower triangular part
	cublasSafeCall(cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, d_A, N, d_B, N));

	// --- Upper triangular part
	cublasSafeCall(cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, d_A, N, d_B, N));

	gpuErrchk(cudaMemcpy(h_B,d_B,N*sizeof(float),cudaMemcpyDeviceToHost));
	
	// --- The output inverted matrix in column-major format
	printf("\n\n");
	for (int i=0; i<N; i++) printf("B[%i]=%f\n", i, h_B[i]);

	return 0;
}
