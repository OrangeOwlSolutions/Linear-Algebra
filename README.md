# Linear-Algebra

The ```Utilities.cu``` and ```Utilities.cuh``` files are mantained at

https://github.com/OrangeOwlSolutions/CUDA_Utilities

and omitted here.

- Null space calculation by the SVD;
- Calculation of the full SVD of a matrix;
- Calculation of the full QR decomposition of a matrix and inversion of a linear system by the QR decomposition approach (see [QR decomposition to solve linear systems in CUDA](http://stackoverflow.com/questions/22399794/qr-decomposition-to-solve-linear-systems-in-cuda));
- Calculation of the LU decomposition of a matrix (see [LU Decomposition in CUDA](http://stackoverflow.com/questions/22814040/lu-decomposition-in-cuda/28799239#28799239));
- Use of LU decomposition to invert linear systems (see [Solving linear systems AX = B with CUDA](http://stackoverflow.com/questions/28794010/solve-ax-b-with-cusolver-library-cuda-7/28799502#28799502));
- Calculation of the determinant of a matrix using Cholesky decomposition and Thrust strided reduction (see [Determinant calculation with CUDA](http://stackoverflow.com/questions/11778981/code-library-to-calculate-determinant-of-a-small-6x6-matrix-solely-on-gpu/29485908#29485908));
- Solve general sparse linear system using cuSOLVER (see [Solving general sparse linear systems in CUDA](http://stackoverflow.com/questions/31840341/solving-general-sparse-linear-systems-in-cuda/32860481#32860481));
- Solve tridiagonal linear systems in CUDA using cuSPARSE (see [Solving tridiagonal linear systems in CUDA](http://stackoverflow.com/questions/19541620/cuda-tridiagonal-solver-seems-not-accelerated-by-gpu/32915677#32915677));
- Solve sparse linear systems by LU factorization (see [Solving sparse linear systems in CUDA using LU factorization](http://stackoverflow.com/questions/17721987/solving-sparse-linear-systems-in-cuda-using-lu-factorization/32916473#32916473));
- Solve sparse positive definite linear systems by Cholesky factorization (see [Solving sparse definite positive linear systems in CUDA](http://stackoverflow.com/questions/30454089/why-does-cusolver-cusolverspdcsrlsvchol-not-work/32927659#32927659));
