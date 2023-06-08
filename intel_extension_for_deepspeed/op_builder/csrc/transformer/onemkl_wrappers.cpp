#include "onemkl_wrappers.hpp"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "conversion_utils.h"

template <typename T>
int onemkl_matmul_ex(sycl::queue handle,
                     oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* A,
                     const T* B,
                     T* C)
{
  // column major
  try {
    int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;
    oneapi::mkl::blas::gemm(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << std::endl;
    std::exit(1);
  }

  return 0;
}

// TODO: if stride_A needed
template <typename T>
int onemkl_strided_batched_gemm(sycl::queue handle,
                                int m,
                                int n,
                                int k,
                                const float alpha,
                                const float beta,
                                const T* A,
                                const T* B,
                                T* C,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch) {
  // column major
  try {
    int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;

    oneapi::mkl::blas::gemm_batch(handle,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  lda,
                                  stride_A,
                                  B,
                                  ldb,
                                  stride_B,
                                  beta,
                                  C,
                                  ldc,
                                  stride_C,
                                  batch);
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << " (batch, m, n, k)" << batch << " " << m << " " << n << " " << k << std::endl;
    std::exit(1);
  }

  return 0;
}

template int onemkl_matmul_ex(sycl::queue handle,
                              oneapi::mkl::transpose transa,
                              oneapi::mkl::transpose transb,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const float* A,
                              const float* B,
                              float* C);

template int onemkl_matmul_ex(sycl::queue handle,
                              oneapi::mkl::transpose transa,
                              oneapi::mkl::transpose transb,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const half* A,
                              const half* B,
                              half* C);

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const float* A,
                                         const float* B,
                                         float* C,
                                         oneapi::mkl::transpose transa,
                                         oneapi::mkl::transpose transb,
                                         int stride_A,
                                         int stride_B,
                                         int stride_C,
                                         int batch);

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const half* A,
                                         const half* B,
                                         half* C,
                                         oneapi::mkl::transpose transa,
                                         oneapi::mkl::transpose transb,
                                         int stride_A,
                                         int stride_B,
                                         int stride_C,
                                         int batch);
