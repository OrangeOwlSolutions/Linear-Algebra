#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "Utilities.cuh"

cusparseHandle_t	handle; 

cusparseMatDescr_t	descrA		= 0;
cusparseMatDescr_t	descr_L		= 0;
cusparseMatDescr_t	descr_U		= 0;

csrilu02Info_t		info_A		= 0;
csrsv2Info_t		info_L		= 0;
csrsv2Info_t		info_U		= 0;

void				*pBuffer	= 0;

/*****************************/
/* SETUP DESCRIPTOR FUNCTION */
/*****************************/
void setUpDescriptor(cusparseMatDescr_t &descrA, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase) {
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, matrixType));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, indexBase));
}

/**************************************************/
/* SETUP DESCRIPTOR FUNCTION FOR LU DECOMPOSITION */
/**************************************************/
void setUpDescriptorLU(cusparseMatDescr_t &descrLU, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase, cusparseFillMode_t fillMode, cusparseDiagType_t diagType) {
	cusparseSafeCall(cusparseCreateMatDescr(&descrLU));
	cusparseSafeCall(cusparseSetMatType(descrLU, matrixType));
	cusparseSafeCall(cusparseSetMatIndexBase(descrLU, indexBase));
	cusparseSafeCall(cusparseSetMatFillMode(descrLU, fillMode));
	cusparseSafeCall(cusparseSetMatDiagType(descrLU, diagType));
}

/**********************************************/
/* MEMORY QUERY FUNCTION FOR LU DECOMPOSITION */
/**********************************************/
void memoryQueryLU(csrilu02Info_t &info_A, csrsv2Info_t &info_L, csrsv2Info_t &info_U, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t descr_L,
				   cusparseMatDescr_t descr_U, double *d_A, int *d_A_RowIndices, int *d_A_ColIndices, cusparseOperation_t matrixOperation, void **pBuffer) {

	cusparseSafeCall(cusparseCreateCsrilu02Info(&info_A));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_U));

	int pBufferSize_M, pBufferSize_L, pBufferSize_U;
	cusparseSafeCall(cusparseDcsrilu02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, &pBufferSize_U));

	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
	gpuErrchk(cudaMalloc((void**)pBuffer, pBufferSize));

}

/******************************************/
/* ANALYSIS FUNCTION FOR LU DECOMPOSITION */
/******************************************/
void analysisLUDecomposition(csrilu02Info_t &info_A, csrsv2Info_t &info_L, csrsv2Info_t &info_U, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t descr_L,
	cusparseMatDescr_t descr_U, double *d_A, int *d_A_RowIndices, int *d_A_ColIndices, cusparseOperation_t matrixOperation, cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2, void *pBuffer) {

	int structural_zero;

	cusparseSafeCall(cusparseDcsrilu02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solvePolicy1, pBuffer));
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }

	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, solvePolicy1, pBuffer));
	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, solvePolicy2, pBuffer));

}

/************************************************/
/* COMPUTE LU DECOMPOSITION FOR SPARSE MATRICES */
/************************************************/
void computeSparseLU(csrilu02Info_t &info_A, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, double *d_A, int *d_A_RowIndices,
	                 int *d_A_ColIndices, cusparseSolvePolicy_t solutionPolicy ,void *pBuffer) {

	int numerical_zero;

	cusparseSafeCall(cusparseDcsrilu02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solutionPolicy, pBuffer));
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); }
	
}

void solveSparseLinearSystem() {


}

/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE
	cusparseSafeCall(cusparseCreate(&handle));

	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
	const int Nrows = 4;                        // --- Number of rows
	const int Ncols = 4;                        // --- Number of columns
	const int N = Nrows;

	// --- Host side dense matrix
	double *h_A_dense = (double*)malloc(Nrows * Ncols * sizeof(*h_A_dense));

	// --- Column-major ordering
	h_A_dense[0] = 0.4612f;  h_A_dense[4] = -0.0006f;   h_A_dense[8] = 0.3566f; h_A_dense[12] = 0.0f;
	h_A_dense[1] = -0.0006f; h_A_dense[5] = 0.4640f;    h_A_dense[9] = 0.0723f; h_A_dense[13] = 0.0f;
	h_A_dense[2] = 0.3566f;  h_A_dense[6] = 0.0723f;    h_A_dense[10] = 0.7543f; h_A_dense[14] = 0.0f;
	h_A_dense[3] = 0.f;      h_A_dense[7] = 0.0f;       h_A_dense[11] = 0.0f;    h_A_dense[15] = 0.1f;

	// --- Create device array and copy host array to it
	double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

	// --- Allocating and defining dense host and device data vectors
	double *h_x = (double *)malloc(Nrows * sizeof(double));
	h_x[0] = 100.0;  h_x[1] = 200.0; h_x[2] = 400.0; h_x[3] = 500.0;

	double *d_x;        gpuErrchk(cudaMalloc(&d_x, Nrows * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_x, h_x, Nrows * sizeof(double), cudaMemcpyHostToDevice));

	/*******************************/
	/* FROM DENSE TO SPARSE MATRIX */
	/*******************************/
	// --- Descriptor for sparse matrix A
	setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE);

	int nnz = 0;                                // --- Number of nonzero elements in dense matrix
	const int lda = Nrows;                      // --- Leading dimension of dense matrix
	// --- Device side number of nonzero elements per row
	int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
	// --- Compute the number of nonzero elements per row and the total number of nonzero elements in the dense d_A_dense
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	// --- Host side number of nonzero elements per row
	int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));
	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

	printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
	for (int i = 0; i < Nrows; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
	printf("\n");

	// --- Device side sparse matrix
	double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
	int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
	int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

	// --- Host side sparse matrix
	double *h_A = (double *)malloc(nnz * sizeof(*h_A));
	int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

	printf("\nOriginal matrix in CSR format\n\n");
	for (int i = 0; i < nnz; ++i) printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

	/******************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR L AND U */
	/******************************************/
	setUpDescriptorLU(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
	setUpDescriptorLU(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);

	/**************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/**************************************************************************************************/
	memoryQueryLU(info_A, info_L, info_U, handle, N, nnz, descrA, descr_L, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE, &pBuffer);

	/************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/************************************************************************************************/
	analysisLUDecomposition(info_A, info_L, info_U, handle, N, nnz, descrA, descr_L, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_SOLVE_POLICY_NO_LEVEL, 
							CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

	/************************************/
	/* STEP 4: FACTORIZATION: A = L * U */
	/************************************/
	computeSparseLU(info_A, handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

	/*********************/
	/* STEP 5: L * z = x */
	/*********************/
	// --- Allocating the intermediate result vector
	double *d_z;        gpuErrchk(cudaMalloc(&d_z, N * sizeof(double)));

	const double alpha = 1.;
	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	/*********************/
	/* STEP 5: U * y = z */
	/*********************/
	// --- Allocating the result vector
	double *d_y;        gpuErrchk(cudaMalloc(&d_y, Ncols * sizeof(double)));

	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	/********************************/
	/* MOVE THE RESULTS TO THE HOST */
	/********************************/
	double *h_y = (double *)malloc(Ncols * sizeof(double));
	gpuErrchk(cudaMemcpy(h_x, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\nFinal result\n");
	for (int k = 0; k<N; k++) printf("x[%i] = %f\n", k, h_x[k]);
}
