#ifndef SOLVESPARSESYSTEMLU_CUH
#define SOLVESPARSESYSTEMLU_CUH

// --- Set up descriptor function
void setUpDescriptor(cusparseMatDescr_t &, cusparseMatrixType_t, cusparseIndexBase_t);

//// --- Set up descriptor function for LU decomposition
//void setUpDescriptorLU(cusparseMatDescr_t &, cusparseMatrixType_t, cusparseIndexBase_t, cusparseFillMode_t, cusparseDiagType_t);
//
//// --- Memory query function for LU decomposition
//void memoryQueryLU(const int, const int, float  * __restrict__, const int * __restrict__, const int * __restrict__, const cusparseOperation_t);
//void memoryQueryLU(const int, const int, double * __restrict__, const int * __restrict__, const int * __restrict__, const cusparseOperation_t);
//
//// --- Analysis function for LU decomposition
//void analysisLUDecomposition(const int, const int, float * __restrict__, const int * __restrict__, const int * __restrict__, cusparseOperation_t, cusparseSolvePolicy_t,
//	cusparseSolvePolicy_t);
//void analysisLUDecomposition(const int, const int, double * __restrict__, const int * __restrict__, const int * __restrict__, cusparseOperation_t, cusparseSolvePolicy_t,
//	cusparseSolvePolicy_t);
//
//// --- Compute LU decomposition for sparse matrices
//void computeSparseLU(const int, const int, float *, int *, int *, cusparseSolvePolicy_t);
//void computeSparseLU(const int, const int, double *, int *, int *, cusparseSolvePolicy_t);

// --- Solve sparse linear system by LU decomposition function
void solveSparseLinearSystemLU(const int * __restrict__, const int * __restrict__, float  * __restrict__, const float  * __restrict__, float  * __restrict__, const int,
	                           const int, cusparseIndexBase_t);
void solveSparseLinearSystemLU(const int * __restrict__, const int * __restrict__, double * __restrict__, const double * __restrict__, double * __restrict__, const int,
							   const int, cusparseIndexBase_t);

#endif
