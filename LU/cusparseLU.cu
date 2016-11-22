#include <stdio.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "Utilities.cuh"

cusparseHandle_t	handle;

cusparseMatDescr_t	descrA = 0;
cusparseMatDescr_t	descr_L = 0;
cusparseMatDescr_t	descr_U = 0;

csrilu02Info_t		info_A = 0;
csrsv2Info_t		info_L = 0;
csrsv2Info_t		info_U = 0;

void				*pBuffer = 0;

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
void memoryQueryLU(const int N, const int nnz, float * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, const cusparseOperation_t matrixOperation) {

	cusparseSafeCall(cusparseCreateCsrilu02Info(&info_A));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_U));

	int pBufferSize_M, pBufferSize_L, pBufferSize_U;
	cusparseSafeCall(cusparseScsrilu02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, &pBufferSize_U));

	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
	gpuErrchk(cudaMalloc(&pBuffer, pBufferSize));

}

void memoryQueryLU(const int N, const int nnz, double * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, const cusparseOperation_t matrixOperation) {

	cusparseSafeCall(cusparseCreateCsrilu02Info(&info_A));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_U));

	int pBufferSize_M, pBufferSize_L, pBufferSize_U;
	cusparseSafeCall(cusparseDcsrilu02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, &pBufferSize_U));

	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
	gpuErrchk(cudaMalloc(&pBuffer, pBufferSize));

}

/******************************************/
/* ANALYSIS FUNCTION FOR LU DECOMPOSITION */
/******************************************/
void analysisLUDecomposition(const int N, const int nnz, float * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, cusparseOperation_t matrixOperation, cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2) {

	int structural_zero;

	cusparseSafeCall(cusparseScsrilu02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solvePolicy1, pBuffer));
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }

	cusparseSafeCall(cusparseScsrsv2_analysis(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, solvePolicy1, pBuffer));
	cusparseSafeCall(cusparseScsrsv2_analysis(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, solvePolicy2, pBuffer));

}

void analysisLUDecomposition(const int N, const int nnz, double * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, cusparseOperation_t matrixOperation, cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2) {

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
void computeSparseLU(const int N, const int nnz, float * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, cusparseSolvePolicy_t solutionPolicy) {

	int numerical_zero;

	cusparseSafeCall(cusparseScsrilu02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solutionPolicy, pBuffer));
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); }

}

void computeSparseLU(const int N, const int nnz, double * __restrict__ d_A, const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, cusparseSolvePolicy_t solutionPolicy) {

	int numerical_zero;

	cusparseSafeCall(cusparseDcsrilu02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solutionPolicy, pBuffer));
	cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); }

}

/***********************************************************/
/* SOLVE SPARSE LINEAR SYSTEM BY LU DECOMPOSITION FUNCTION */
/***********************************************************/
void solveSparseLinearSystemLU(const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, float * __restrict__ d_A,
	                           const float * __restrict__ d_x, float * __restrict__ d_y, const int nnz, const int Nrows, cusparseIndexBase_t indexBase)
{
	// --- Initialize cuSPARSE
	cusparseSafeCall(cusparseCreate(&handle));

	const int Ncols = Nrows;                    // --- Number of columns
	const int N		= Nrows;
	const int lda	= Nrows;                    // --- Leading dimension of dense matrix

	/*********************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR A, L AND U */
	/*********************************************/
	setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase);
	setUpDescriptorLU(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
	setUpDescriptorLU(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);

	/**************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/**************************************************************************************************/
	memoryQueryLU(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE);

	/************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/************************************************************************************************/
	analysisLUDecomposition(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_SOLVE_POLICY_NO_LEVEL, CUSPARSE_SOLVE_POLICY_USE_LEVEL);

	/************************************/
	/* STEP 4: FACTORIZATION: A = L * U */
	/************************************/
	computeSparseLU(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_SOLVE_POLICY_NO_LEVEL);

	/*********************/
	/* STEP 5: L * z = x */
	/*********************/
	// --- Allocating the intermediate result vector
	float *d_z;        gpuErrchk(cudaMalloc(&d_z, N * sizeof(float)));

	const float alpha = 1.;
	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	/*********************/
	/* STEP 5: U * y = z */
	/*********************/
	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

}

void solveSparseLinearSystemLU(const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, double * __restrict__ d_A,
							   const double * __restrict__ d_x, double * __restrict__ d_y, const int nnz, const int Nrows, cusparseIndexBase_t indexBase)
{
	// --- Initialize cuSPARSE
	cusparseSafeCall(cusparseCreate(&handle));

	const int Ncols = Nrows;                    // --- Number of columns
	const int N = Nrows;
	const int lda = Nrows;                    // --- Leading dimension of dense matrix

	/*********************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR A, L AND U */
	/*********************************************/
	setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase);
	setUpDescriptorLU(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
	setUpDescriptorLU(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL, indexBase, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);

	/**************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/**************************************************************************************************/
	memoryQueryLU(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE);

	/************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: LU FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/************************************************************************************************/
	analysisLUDecomposition(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_SOLVE_POLICY_NO_LEVEL, CUSPARSE_SOLVE_POLICY_USE_LEVEL);

	/************************************/
	/* STEP 4: FACTORIZATION: A = L * U */
	/************************************/
	computeSparseLU(N, nnz, d_A, d_A_RowIndices, d_A_ColIndices, CUSPARSE_SOLVE_POLICY_NO_LEVEL);

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
	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

}
