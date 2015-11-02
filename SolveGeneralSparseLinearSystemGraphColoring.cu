#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#define BLOCKSIZE	256

/**************************/
/* SETTING UP THE PROBLEM */
/**************************/
void setUpTheProblem(double **h_A_dense, double **h_x_dense, double **d_A_dense, double **d_x_dense, const int N) {

	// --- Host side dense matrix
	h_A_dense[0] = (double*)calloc(N * N, sizeof(*h_A_dense));
	
	// --- Column-major ordering
	h_A_dense[0][0] = 0.4612f;  h_A_dense[0][4] = -0.0006f;	h_A_dense[0][8]  = 0.f; h_A_dense[0][12] = 0.0f; 
	h_A_dense[0][1] = -0.0006f; h_A_dense[0][5] = 0.f;	h_A_dense[0][9]  = 0.0723f; h_A_dense[0][13] = 0.04f; 
	h_A_dense[0][2] = 0.3566f;  h_A_dense[0][6] = 0.0723f;	h_A_dense[0][10] = 0.f; h_A_dense[0][14] = 0.0f; 
	h_A_dense[0][3] = 0.0f;	    h_A_dense[0][7] = 0.0f;		h_A_dense[0][11] = 1.0f;	h_A_dense[0][15] = 0.1f; 

	h_x_dense[0]	= (double *)malloc(N * sizeof(double)); 
	h_x_dense[0][0] = 100.0;  h_x_dense[0][1] = 200.0; h_x_dense[0][2] = 400.0; h_x_dense[0][3] = 500.0;

	// --- Create device arrays and copy host arrays to them
	gpuErrchk(cudaMalloc(&d_A_dense[0], N * N * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A_dense[0], h_A_dense[0], N * N * sizeof(double), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_x_dense[0], N * sizeof(double)));   
    gpuErrchk(cudaMemcpy(d_x_dense[0], h_x_dense[0], N * sizeof(double), cudaMemcpyHostToDevice));
}

/************************/
/* FROM DENSE TO SPARSE */
/************************/
void fromDenseToSparse(const cusparseHandle_t handle, double *d_A_dense, double **d_A, int **d_A_RowIndices, int **d_A_ColIndices, int *nnz, 
	                   cusparseMatDescr_t *descrA, const int N) {
	
	cusparseSafeCall(cusparseCreateMatDescr(&descrA[0]));
	cusparseSafeCall(cusparseSetMatType		(descrA[0], CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA[0], CUSPARSE_INDEX_BASE_ZERO));  
	
	nnz[0] = 0;								// --- Number of nonzero elements in dense matrix
	const int lda = N;						// --- Leading dimension of dense matrix
	
	// --- Device side number of nonzero elements per row
	int *d_nnzPerVector; 	gpuErrchk(cudaMalloc(&d_nnzPerVector, N * sizeof(int)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA[0], d_A_dense, lda, d_nnzPerVector, &nnz[0]));
	
	// --- Host side number of nonzero elements per row
	int *h_nnzPerVector = (int *)malloc(N * sizeof(int));
	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, N * sizeof(int), cudaMemcpyDeviceToHost));

	printf("Number of nonzero elements in dense matrix = %i\n\n", nnz[0]);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
	printf("\n");

	// --- Device side sparse matrix
	gpuErrchk(cudaMalloc(&d_A[0], nnz[0] * sizeof(double)));
	
	gpuErrchk(cudaMalloc(&d_A_RowIndices[0], (N + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_A_ColIndices[0], nnz[0]  * sizeof(int)));
	
	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA[0], d_A_dense, lda, d_nnzPerVector, d_A[0], d_A_RowIndices[0], d_A_ColIndices[0]));

	// --- Host side sparse matrix
	double *h_A = (double *)malloc(nnz[0] * sizeof(double));		
	int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnz[0] * sizeof(*h_A_ColIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A[0], nnz[0] * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices[0], (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices[0], nnz[0] * sizeof(int), cudaMemcpyDeviceToHost));
	
	printf("\nOriginal matrix in CSR format\n\n");
	for (int i = 0; i < nnz[0]; ++i) printf("A[%i] = %f ", i, h_A[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	for (int i = 0; i < nnz[0]; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);	
	
}

/******************/
/* GRAPH COLORING */
/******************/
__global__ void setRowIndices(int *d_B_RowIndices, const int N) {

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid == N)		d_B_RowIndices[tid] = N;
	else if (tid < N)	d_B_RowIndices[tid] = tid;

}

__global__ void setB(double *d_B, const int N) {

	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < N)	d_B[tid] = 1.f;

}

void graphColoring(const cusparseHandle_t handle, const int nnz, const cusparseMatDescr_t descrA, const double fractionToColor, double *d_A, 
				   const int *d_A_RowIndices, const int *d_A_ColIndices, double **d_B, int **d_B_RowIndices, int **d_B_ColIndices, 
				   cusparseMatDescr_t *descrB, const int N) {

	cusparseColorInfo_t info;		cusparseSafeCall(cusparseCreateColorInfo(&info));

	int ncolors;
	int *d_coloring;		gpuErrchk(cudaMalloc(&d_coloring, N * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_B_ColIndices[0], N * sizeof(double)));
	cusparseSafeCall(cusparseDcsrcolor(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, &fractionToColor, &ncolors, d_coloring,
									   d_B_ColIndices[0], info));

	int *h_coloring		= (int *)malloc(N * sizeof(double));
	int *h_B_ColIndices	= (int *)malloc(N * sizeof(double));
	gpuErrchk(cudaMemcpy(h_coloring, d_coloring, N * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices[0], N * sizeof(double), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) printf("h_coloring = %i; h_B_ColIndices = %i\n", h_coloring[i], h_B_ColIndices[i]);

	gpuErrchk(cudaMalloc(&d_B_RowIndices[0], (N + 1) * sizeof(int)));
	int *h_B_RowIndices	= (int *)malloc((N + 1) * sizeof(double));
	setRowIndices<<<iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE>>>(d_B_RowIndices[0], N);

	gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices[0], (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	printf("\n"); for (int i = 0; i <= N; i++) printf("h_B_RowIndices = %i\n", h_B_RowIndices[i]);

	gpuErrchk(cudaMalloc(&d_B[0], N * sizeof(double)));
	double *h_B	= (double *)malloc(N * sizeof(double));
	setB<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_B[0], N);

	gpuErrchk(cudaMemcpy(h_B, d_B[0], N * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n"); for (int i = 0; i < N; i++) printf("h_B = %f\n", h_B[i]);

	// --- Descriptor for sparse mutation matrix B
	cusparseSafeCall(cusparseCreateMatDescr(&descrB[0]));
	cusparseSafeCall(cusparseSetMatType		(descrB[0], CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrB[0], CUSPARSE_INDEX_BASE_ZERO));  
}

/*************************/
/* MATRIX ROW REORDERING */
/*************************/
void matrixRowReordering(const cusparseHandle_t handle, int nnzA, int nnzB, int *nnzC, cusparseMatDescr_t descrA, cusparseMatDescr_t descrB, 
	                     cusparseMatDescr_t *descrC, double *d_A, int *d_A_RowIndices, int *d_A_ColIndices, double *d_B, int *d_B_RowIndices, 
						 int *d_B_ColIndices, double **d_C, int **d_C_RowIndices, int **d_C_ColIndices, const int N) {

	// --- Descriptor for sparse matrix C
	cusparseSafeCall(cusparseCreateMatDescr(&descrC[0]));
	cusparseSafeCall(cusparseSetMatType		(descrC[0], CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrC[0], CUSPARSE_INDEX_BASE_ZERO));  

	const int lda = N;						// --- Leading dimension of dense matrix

	// --- Device side sparse matrix
	gpuErrchk(cudaMalloc(&d_C_RowIndices[0], (N + 1) * sizeof(int)));
	
	// --- Host side sparse matrices
	int *h_C_RowIndices = (int *)malloc((N + 1) * sizeof(int));
	
	// --- Performing the matrix - matrix multiplication
	int baseC;
	int *nnzTotalDevHostPtr = &nnzC[0];	
	
	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB, 
										 d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC[0], d_C_RowIndices[0], 
										 nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) nnzC[0] = *nnzTotalDevHostPtr;
	else {
		gpuErrchk(cudaMemcpy(&nnzC[0],  d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC,    d_C_RowIndices,     sizeof(int), cudaMemcpyDeviceToHost));
		nnzC -= baseC;
	}
	gpuErrchk(cudaMalloc(&d_C_ColIndices[0], nnzC[0] * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_C[0], nnzC[0] * sizeof(double)));
	double *h_C = (double *)malloc(nnzC[0] * sizeof(double));		
	int *h_C_ColIndices = (int *)malloc(nnzC[0] * sizeof(int));
	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB,
									  d_B, d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC[0],
									  d_C[0], d_C_RowIndices[0], d_C_ColIndices[0]));

	double *h_C_dense = (double*)malloc(N * N * sizeof(double));
	double *d_C_dense;	gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(double)));
	cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC[0], d_C[0], d_C_RowIndices[0], d_C_ColIndices[0], d_C_dense, N));

	gpuErrchk(cudaMemcpy(h_C ,           d_C[0],            nnzC[0] * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices[0], (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices[0], nnzC[0] * sizeof(int), cudaMemcpyDeviceToHost));
	
	printf("\nResult matrix C in CSR format\n\n");
	for (int i = 0; i < nnzC[0]; ++i) printf("C[%i] = %f ", i, h_C[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < nnzC[0]; ++i) printf("h_C_ColIndices[%i] = %i \n", i, h_C_ColIndices[i]);	
	
	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) 
			printf("%f \t", h_C_dense[i * N + j]);
		printf("\n");
		}

}

/******************/
/* ROW REORDERING */
/******************/
void rowReordering(const cusparseHandle_t handle, int nnzA, cusparseMatDescr_t descrB, double *d_B, int *d_B_RowIndices, int *d_B_ColIndices, 
	               double *d_x_dense, double **d_y_dense, const int N) {

	gpuErrchk(cudaMalloc(&d_y_dense[0], N     * sizeof(double)));

	const double alpha = 1.;
	const double beta  = 0.;
	cusparseSafeCall(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzA, &alpha, descrB, d_B, d_B_RowIndices, d_B_ColIndices, d_x_dense, 
		                            &beta, d_y_dense[0]));

	double *h_y_dense = (double*)malloc(N *     sizeof(double));
	gpuErrchk(cudaMemcpy(h_y_dense,           d_y_dense[0],            N * sizeof(double), cudaMemcpyDeviceToHost));
		
	printf("\nResult vector\n\n");
	for (int i = 0; i < N; ++i) printf("h_y[%i] = %f ", i, h_y_dense[i]); printf("\n");

}

/*****************************/
/* SOLVING THE LINEAR SYSTEM */
/*****************************/
void LUDecomposition(const cusparseHandle_t handle, int nnzC, cusparseMatDescr_t descrC, double *d_C, int *d_C_RowIndices, int *d_C_ColIndices, 
	                 double *d_x_dense, double **d_y_dense, const int N) {

	/******************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR L AND U */
	/******************************************/
	cusparseMatDescr_t		descr_L = 0; 
	cusparseSafeCall(cusparseCreateMatDescr	(&descr_L)); 
	cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO)); 
	cusparseSafeCall(cusparseSetMatType		(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL)); 
	cusparseSafeCall(cusparseSetMatFillMode	(descr_L, CUSPARSE_FILL_MODE_LOWER)); 
	cusparseSafeCall(cusparseSetMatDiagType	(descr_L, CUSPARSE_DIAG_TYPE_UNIT)); 
	
	cusparseMatDescr_t		descr_U = 0; 
	cusparseSafeCall(cusparseCreateMatDescr	(&descr_U)); 
	cusparseSafeCall(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO)); 
	cusparseSafeCall(cusparseSetMatType		(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL)); 
	cusparseSafeCall(cusparseSetMatFillMode	(descr_U, CUSPARSE_FILL_MODE_UPPER)); 
	cusparseSafeCall(cusparseSetMatDiagType	(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT)); 
	
	/**************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/**************************************************************************************************/
	csrilu02Info_t info_C = 0; cusparseSafeCall(cusparseCreateCsrilu02Info	(&info_C)); 
	csrsv2Info_t info_L = 0;   cusparseSafeCall(cusparseCreateCsrsv2Info	(&info_L)); 
	csrsv2Info_t info_U = 0;   cusparseSafeCall(cusparseCreateCsrsv2Info	(&info_U)); 
	
	int pBufferSize_M, pBufferSize_L, pBufferSize_U; 
	cusparseSafeCall(cusparseDcsrilu02_bufferSize(handle, N, nnzC, descrC, d_C, d_C_RowIndices, d_C_ColIndices, info_C, &pBufferSize_M)); 
	cusparseSafeCall(cusparseDcsrsv2_bufferSize	(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, descr_L, d_C, d_C_RowIndices, d_C_ColIndices, info_L, &pBufferSize_L)); 
	cusparseSafeCall(cusparseDcsrsv2_bufferSize	(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, descr_U, d_C, d_C_RowIndices, d_C_ColIndices, info_U, &pBufferSize_U)); 
	
	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U)); 
	void *pBuffer = 0; gpuErrchk(cudaMalloc((void**)&pBuffer, pBufferSize)); 
	
	/************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/************************************************************************************************/
	int structural_zero; 

	cusparseSafeCall(cusparseDcsrilu02_analysis(handle, N, nnzC, descrC, d_C, d_C_RowIndices, d_C_ColIndices, info_C, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer)); 
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_C, &structural_zero); 
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); } 
	
	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, descr_L, d_C, d_C_RowIndices, d_C_ColIndices, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer)); 
	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, descr_U, d_C, d_C_RowIndices, d_C_ColIndices, info_U, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer)); 
	
	/************************************/
	/* STEP 4: FACTORIZATION: A = L * U */
	/************************************/
	int numerical_zero; 

	cusparseSafeCall(cusparseDcsrilu02(handle, N, nnzC, descrC, d_C, d_C_RowIndices, d_C_ColIndices, info_C, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer)); 
	status = cusparseXcsrilu02_zeroPivot(handle, info_C, &numerical_zero); 
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); } 
	
	/*********************/
	/* STEP 5: L * z = x */
	/*********************/
	// --- Allocating the intermediate result vector
	double *d_z_dense;		gpuErrchk(cudaMalloc(&d_z_dense, N * sizeof(double))); 
	   
	const double alpha = 1.; 
	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, &alpha, descr_L, d_C, d_C_RowIndices, d_C_ColIndices, info_L, d_x_dense, d_z_dense, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer)); 
	
	/*********************/
	/* STEP 5: U * y = z */
	/*********************/
	gpuErrchk(cudaMalloc(&d_y_dense[0], N * sizeof(double))); 
	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnzC, &alpha, descr_U, d_C, d_C_RowIndices, d_C_ColIndices, info_U, d_z_dense, d_y_dense[0], CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	double *h_y_dense = (double *)malloc(N * sizeof(double));
	gpuErrchk(cudaMemcpy(h_y_dense, d_y_dense[0], N * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\nFinal result\n");
	for (int k=0; k<N; k++) printf("x[%i] = %f\n", k, h_y_dense[k]);

}

/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE
	cusparseHandle_t handle;	cusparseSafeCall(cusparseCreate(&handle));

	/*************************************************/
	/* SETTING UP THE ORIGINAL LINEAR SYSTEM PROBLEM */
	/*************************************************/
	const int N     = 4;				// --- Number of rows and columns

	double *h_A_dense;	double *h_x_dense;
	double *d_A_dense;	double *d_x_dense;
	setUpTheProblem(&h_A_dense, &h_x_dense, &d_A_dense, &d_x_dense, N);
		
	/************************/
	/* FROM DENSE TO SPARSE */
	/************************/
	//--- Descriptor for sparse matrix A
	cusparseMatDescr_t descrA;

	int *d_A_RowIndices, *d_A_ColIndices;	
	double *d_A;
	
	int nnzA;
	
	fromDenseToSparse(handle, d_A_dense, &d_A, &d_A_RowIndices, &d_A_ColIndices, &nnzA, &descrA, N);
	
	/******************/
	/* GRAPH COLORING */
	/******************/
	const double fractionToColor = 0.95;
	
	int *d_B_RowIndices, *d_B_ColIndices;	
	double *d_B;

	int nnzB;
	
	cusparseMatDescr_t descrB;		
	graphColoring(handle, nnzB, descrA, fractionToColor, d_A, d_A_RowIndices, d_A_ColIndices, &d_B, &d_B_RowIndices, &d_B_ColIndices, &descrB, N);

	/*************************/
	/* MATRIX ROW REORDERING */
	/*************************/
	int nnzC;
	
	int *d_C_RowIndices, *d_C_ColIndices;
	double *d_C;

	cusparseMatDescr_t descrC;
	matrixRowReordering(handle, nnzA, nnzB, &nnzC, descrA, descrB, &descrC, d_A, d_A_RowIndices, d_A_ColIndices, d_B, d_B_RowIndices, d_B_ColIndices, 
						&d_C, &d_C_RowIndices, &d_C_ColIndices, N);

	/******************/
	/* ROW REORDERING */
	/******************/
	double *d_y_dense;
	rowReordering(handle, nnzA, descrB, d_B, d_B_RowIndices, d_B_ColIndices, d_x_dense, &d_y_dense, N);

	/*****************************/
	/* SOLVING THE LINEAR SYSTEM */
	/*****************************/
	double *d_xsol_dense;
	LUDecomposition(handle, nnzC, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_y_dense, &d_xsol_dense, N);

}
