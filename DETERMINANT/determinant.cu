#include "cuda_runtime.h"
#include "device_launch_paraMeters.h"

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<ostream>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/copy.h>

/*************************/
/* STRIDED RANGE FUNCTOR */
/*************************/
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

int main(void)
{
    const int Nrows = 5;
    const int Ncols = 5;

	const int STRIDE = Nrows + 1;

    // --- Setting the host, Nrows x Ncols matrix
    double h_A[Nrows][Ncols] = { 
        { 2.,    -2.,    -2.,    -2.,    -2.,},  
        {-2.,     4.,     0.,     0.,     0.,}, 
        {-2.,     0.,     6.,     2.,     2.,}, 
        {-2.,     0.,     2.,     8.,     4.,}, 
        {-2.,     0.,     2.,     4.,     10.,}
    };

    // --- Setting the device matrix and moving the host matrix to the device
    double *d_A;            gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;           gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUDA CHOLESKY initialization
    cusolveSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, &work_size));

    // --- CUDA POTRF execution
    double *work;   gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));
    cusolveSafeCall(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, work, work_size, devInfo));
    int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) std::cout   << "Unsuccessful potrf execution\n\n";

    cusolverDnDestroy(solver_handle);

    // --- Strided reduction of the elements of d_A: calculating the product of the diagonal of the Cholesky factorization	
	thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(d_A);
	typedef thrust::device_vector<double>::iterator Iterator;
    strided_range<Iterator> pos(dev_ptr, dev_ptr + Nrows * Ncols, STRIDE);

	double det = thrust::reduce(pos.begin(), pos.end(), 1., thrust::multiplies<double>());
	det  = det * det;

    printf("determinant = %f\n", det);

    return 0;
}
