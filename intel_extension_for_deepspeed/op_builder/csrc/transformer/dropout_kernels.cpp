#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif
#include "custom_sycl_layers.hpp"
#include "conversion_utils.h"

const int unroll_factor = 4;

template <typename T>
void dropout_kernel(const int N,
                    const float ratio,
                    T* out,
                    const T* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    oneapi::mkl::rng::device::philox4x32x10<1> engine(seed.first, {idx, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float rand = oneapi::mkl::rng::device::generate(distr, engine);
        uint8_t m;

        m = (uint8_t)(rand > ratio);

        int i = j;

        mask[i] = (uint8_t)m;

        float input_f = conversion::to<float>(Xdata[i]);
        out[i] = conversion::to<T>(input_f * scale * m);
    }
}

template <typename T>
void dropout_kernel_bwd(const int N,
                        const float ratio,
                        const T* Xdata,
                        T* out,
                        uint8_t* mask,
                        const std::pair<uint64_t, uint64_t>& seed,
                        nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        int i = j;

        float input_f = conversion::to<float>(Xdata[i]);
        out[i] = conversion::to<T>(mask[i] * input_f * scale);
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    queue stream,
                    bool bwd)
{
    /*
     * dropout.Forward
     */
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    if (dim > 512) {
        block_dim[2] >>= 1;
        grid_dim[2] <<= 1;
    }
    uint64_t inc = total_count / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);
    if (bwd)
        stream.submit([&](handler& cgh) {
            cgh.parallel_for(
                nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
                    dropout_kernel_bwd(total_count, ratio, vals, out, mask, seed, item_ct1);
                });
        });
    else
        stream.submit([&](handler& cgh) {
            cgh.parallel_for(
                nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
                    dropout_kernel(total_count, ratio, out, vals, mask, seed, item_ct1);
                });
        });
}

template void launch_dropout(float* out,
                             const float* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue stream,
                             bool);
template void launch_dropout(bf16* out,
                             const bf16* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue stream,
                             bool);
template void launch_dropout(half* out,
                             const half* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             queue stream,
                             bool);


template <typename T>
void dropout_grad_kernel(const int N,
                         const float scale,
                         T* Xdata,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    DPCPP_1D_KERNEL_LOOP(i, N)
    {
        float input_f = conversion::to<float>(Xdata[i]);
        Xdata[i] = conversion::to<T>(input_f * scale * mask[i]);
    }
}

template <typename T>
void launch_dropout_grad(T* vals, uint8_t* mask, int total_count, float ratio, queue stream)
{
    /*
     * Dropout.Backward0
     */
    const float scale = 1. / (1. - ratio);
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);
    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_grad_kernel(total_count, scale, vals, mask, item_ct1);
        });
    });
}

template void launch_dropout_grad(float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);
template void launch_dropout_grad(bf16* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);
template void launch_dropout_grad(half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);

template <typename T>
void dropout_grad_kernel(const int N,
                         const float scale,
                         const T* Xdata,
                         T* out,
                         uint8_t* mask,
                         nd_item<3> item_ct1)
{
    DPCPP_1D_KERNEL_LOOP(i, N)
    {
        float input_f = conversion::to<float>(Xdata[i]);
        out[i] = conversion::to<T>(input_f * scale * mask[i]);
    }
}

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         queue stream)
{
    /*
     * Dropout.Backward1
     */
    const float scale = 1. / (1. - ratio);
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);
    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_grad_kernel(total_count, scale, vals, vals_out, mask, item_ct1);
        });
    });
}
template void launch_dropout_grad(float* vals_out,
                                  const float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);
template void launch_dropout_grad(bf16* vals_out,
                                  const bf16* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);
template void launch_dropout_grad(half* vals_out,
                                  const half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  queue stream);

template <typename T>
void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const T* bias,
                    T* Xdata,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim);

    oneapi::mkl::rng::device::philox4x32x10<1> engine(seed.first, {idx, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;


    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float rand = oneapi::mkl::rng::device::generate(distr, engine);
        
        uint8_t m;
        m = (uint8_t)(rand > ratio);

        float bias_f = conversion::to<float>(bias[j % dim]);
        float input_f = conversion::to<float>(Xdata[j]);

        float output_f = bias_f + input_f;
        output_f = output_f * scale * m;

        mask[j] = m;
        Xdata[j] = conversion::to<T>(output_f);
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    queue stream)
{
    int total_count = batch * dim;

    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    uint64_t inc = (batch * dim) / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);
    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_kernel(total_count, dim, ratio, bias, out, mask, seed, item_ct1);
        });
    });
}

template void launch_dropout(float*,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue stream);
template void launch_dropout(half*,
                             const half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue stream);

template <typename T>
void dropout_kernel(const int N,
                    const int dim,
                    const float ratio,
                    const T* input,
                    const T* residual,
                    const T* bias,
                    T* out,
                    uint8_t* mask,
                    const std::pair<uint64_t, uint64_t>& seed,
                    nd_item<3> item_ct1)
{
    const float scale = 1. / (1. - ratio);
    size_t idx =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    int tid = item_ct1.get_local_id(2) % (dim);

    oneapi::mkl::rng::device::philox4x32x10<1> engine(seed.first, {idx, seed.second});
    oneapi::mkl::rng::device::uniform<> distr;

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float rand = oneapi::mkl::rng::device::generate(distr, engine);

        uint8_t m;

        m = (uint8_t)(rand > ratio);

        float bias_f = conversion::to<float>(bias[j % dim]);
        float residual_f = conversion::to<float>(residual[j]);
        float input_f = conversion::to<float>(input[j]);

        float output_f = bias_f + input_f;
        output_f = output_f * scale * m;
        output_f += residual_f;
        
        mask[j] = m;
        out[j] = conversion::to<T>(output_f);
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* input,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    queue stream)
{
    int total_count = batch * dim;
    range<3> grid_dim = range<3>(1, 1, DS_GET_BLOCKS(total_count));
    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);

    uint64_t inc = (batch * dim) / grid_dim[2] / block_dim[2];
    std::pair<uint64_t, uint64_t> seed = SyclContext::Instance().IncrementOffset(inc);

    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim), [=](nd_item<3> item_ct1) {
            dropout_kernel(
                total_count, dim, ratio, input, residual, bias, out, mask, seed, item_ct1);
        });
    });
}

template void launch_dropout(float*,
                             const float*,
                             const float* residual,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue stream);
template void launch_dropout(bf16*,
                             const bf16*,
                             const bf16* residual,
                             const bf16* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue stream);
template void launch_dropout(half*,
                             const half*,
                             const half* residual,
                             const half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             queue stream);
