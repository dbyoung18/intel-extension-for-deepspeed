#include "common.hpp"
#include "context.hpp"
#include "conversion_utils.h"
#include "custom_sycl_layers.hpp"
#include "dropout.hpp"
#include "feed_forward.hpp"
#include "gelu.hpp"
#include "general_kernels.hpp"
#include "normalize_layer.hpp"
#include "softmax.hpp"
#include "strided_batch_gemm.hpp"

template <typename T>
std::vector<torch::Tensor> dropout_forward(float ratio,
                                           uint32_t dim,
                                           int bsz,
                                           const torch::Tensor& vals)
{
    CHECK_INPUT(vals);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .layout(torch::kStrided)
                             .device(torch::kXPU)
                             .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T* input_ptr = (const T*)vals.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.Forward(bsz, output_ptr, input_ptr, q);
    return {output, mask};
}

template <typename T>
std::vector<torch::Tensor> dropout_forward_with_bias(float ratio,
                                                     uint32_t dim,
                                                     int bsz,
                                                     const torch::Tensor& vals,
                                                     const torch::Tensor& bias,
                                                     const torch::Tensor& residual)
{
    CHECK_INPUT(vals);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
                             .dtype(torch::kInt8)
                             .layout(torch::kStrided)
                             .device(torch::kXPU)
                             .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T* input_ptr = (const T*)vals.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    const T* residual_ptr = (const T*)residual.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.ForwardWithBias(bsz, output_ptr, input_ptr, residual_ptr, bias_ptr, q);
    return {output, mask};
}

template <typename T>
std::vector<torch::Tensor> dropout_backward(float ratio,
                                            uint32_t dim,
                                            int bsz,
                                            torch::Tensor& vals,
                                            torch::Tensor& mask,
                                            bool in_place)
{
    CHECK_INPUT(vals);
    CHECK_INPUT(mask);
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    uint8_t* mask_ptr = (uint8_t*)mask.data_ptr();
    _dropout.SetMask(mask_ptr);
    if (in_place) {
        T* d_input_ptr = (T*)vals.data_ptr();
        _dropout.Backward(bsz, d_input_ptr, q);
        return {vals};
    } else {
        auto output = torch::empty_like(vals);
        const T* d_input_ptr = (const T*)vals.data_ptr();
        T* d_output_ptr = (T*)output.data_ptr();
        _dropout.Backward(bsz, d_output_ptr, d_input_ptr, q);
        return {output};
    }
}

template <typename T>
std::vector<torch::Tensor> feedforward_forward(int bsz,
                                               int seq_len,
                                               int hidden_size,
                                               const torch::Tensor& input,
                                               const torch::Tensor& weights)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    int batchSize = bsz * seq_len;
    int inputSize = hidden_size;
    int outputSize = 3 * hidden_size;
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    const T* input_ptr = (const T*)input.data_ptr();
    const T* weights_ptr = (const T*)weights.data_ptr();

    auto output = torch::empty({bsz, seq_len, outputSize}, options);

    T* output_ptr = (T*)output.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    FeedForward<T> _ff =
        FeedForward<T>(typename FeedForward<T>::Config(batchSize, outputSize, inputSize));

    _ff.Forward(batchSize, input_ptr, weights_ptr, output_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> feedforward_backward(int bsz,
                                                int seq_len,
                                                int hidden_size,
                                                const torch::Tensor& grad_out,
                                                const torch::Tensor& input,
                                                const torch::Tensor& weights)
{
    CHECK_INPUT(grad_out);
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    int batchSize = bsz * seq_len;
    int inputSize = hidden_size;
    int outputSize = 3 * hidden_size;

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    const T* grad_out_ptr = (const T*)grad_out.data_ptr();
    const T* input_ptr = (const T*)input.data_ptr();
    const T* weights_ptr = (const T*)weights.data_ptr();

    auto grad_weights = torch::empty(weights.sizes(), options);
    auto grad_bias = torch::empty({outputSize}, options);
    auto grad_input = torch::empty(input.sizes(), options);

    T* grad_w_ptr = (T*)grad_weights.data_ptr();
    T* grad_b_ptr = (T*)grad_bias.data_ptr();
    T* grad_i_ptr = (T*)grad_input.data_ptr();
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    FeedForward<T> _ff =
        FeedForward<T>(typename FeedForward<T>::Config(batchSize, outputSize, inputSize));

    _ff.Backward(
        batchSize, grad_out_ptr, input_ptr, weights_ptr, grad_w_ptr, grad_b_ptr, q, q, grad_i_ptr);
    return {grad_input, grad_weights, grad_bias};
}

template <typename T>
std::vector<torch::Tensor> transform4d_0213(const torch::Tensor& input,
                                            int batch,
                                            int seq_len,
                                            int hidden_size,
                                            int num_heads,
                                            int trans_count)
{
    CHECK_INPUT(input);
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    torch::Tensor output;
    if (trans_count == 3)
        // trans_count=3
        output = torch::empty({batch, seq_len, 3, num_heads, hidden_size / num_heads}, options);
    else
        // for 1 attn_o_inp, trans_count=1
        output = torch::empty({batch, seq_len, num_heads, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    // trans_count=1
    // launch_transform4d_0213(output_ptr, input_ptr, batch, num_heads, seq_len,
    // hidden_size, q, 1);
    // trans_count=3
    launch_transform4d_0213(
        output_ptr, input_ptr, batch, num_heads, seq_len, hidden_size, q, trans_count);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> bias_add_transform_0213(const torch::Tensor& input,
                                                   const torch::Tensor& bias,
                                                   int batch,
                                                   int seq_len,
                                                   int hidden_size,
                                                   int num_heads)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({3, batch, num_heads, seq_len, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    launch_bias_add_transform_0213(
        output_ptr, input_ptr, bias_ptr, batch, seq_len, hidden_size, num_heads, q, 3);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> transform_0213(const torch::Tensor& input,
                                          int batch,
                                          int seq_len,
                                          int hidden_size,
                                          int num_heads)
{
    CHECK_INPUT(input);

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, num_heads, seq_len, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    T* output_ptr = (T*)output.data_ptr();

    launch_transform_0213(output_ptr, input_ptr, batch, seq_len, hidden_size, num_heads, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> fused_add2(const torch::Tensor& input1,
                                      const torch::Tensor& input2,
                                      int batch,
                                      int seq_len,
                                      int hidden_size)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);

    auto options = torch::TensorOptions()
                       .dtype(input1.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, seq_len, hidden_size}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr1 = (const T*)input1.data_ptr();
    const T* input_ptr2 = (const T*)input2.data_ptr();
    T* output_ptr = (T*)output.data_ptr();

    launch_fused_add2(output_ptr, input_ptr1, input_ptr2, batch, seq_len, hidden_size, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> gelu_forward(int intermediate_size,
                                        int bsz_seq,
                                        const torch::Tensor& input,
                                        const torch::Tensor& bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    auto output = torch::empty_like(input);
    T* output_ptr = (T*)output.data_ptr();
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Gelu<T> _gelu = Gelu<T>(typename Gelu<T>::Config(intermediate_size));
    _gelu.ForwardWithBiasAdd(bsz_seq, input_ptr, bias_ptr, output_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> gelu_backward(torch::Tensor& d_output,
                                         int intermediate_size,
                                         int bsz_seq,
                                         const torch::Tensor& input,
                                         const torch::Tensor& bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    T* d_output_ptr = (T*)d_output.data_ptr();
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Gelu<T> _gelu = Gelu<T>(typename Gelu<T>::Config(intermediate_size));
    _gelu.Backward(bsz_seq, d_output_ptr, input_ptr, bias_ptr, q);
    return {d_output};
}

template <typename T>
std::vector<torch::Tensor> normalize_forward(const int batch,
                                             const int seq_len,
                                             const int hidden_size,
                                             const torch::Tensor& residual,
                                             const torch::Tensor& gamma,
                                             const torch::Tensor& betta,
                                             torch::Tensor& mean,
                                             torch::Tensor& var,
                                             const bool preln,
                                             const bool wmean,
                                             const float epsilon)
{
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);
    CHECK_INPUT(betta);

    int bsz_seq = batch * seq_len;

    auto options = torch::TensorOptions()
                       .dtype(residual.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, seq_len, hidden_size}, options);

    T* output_ptr = (T*)output.data_ptr();
    T* mean_ptr = (T*)mean.data_ptr();
    T* var_ptr = (T*)var.data_ptr();
    const T* residual_ptr = (const T*)residual.data_ptr();
    const T* gamma_ptr = (const T*)gamma.data_ptr();
    const T* betta_ptr = (const T*)betta.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Normalize_Layer<T> _norm(
        typename Normalize_Layer<T>::Config(batch, seq_len, hidden_size, epsilon, true, wmean));
    _norm.SetMean(mean_ptr);
    _norm.SetVar(var_ptr);

    if (wmean)
        _norm.ForwardCheckpoint(bsz_seq, output_ptr, residual_ptr, gamma_ptr, betta_ptr, q);
    else
        _norm.Forward(bsz_seq, output_ptr, residual_ptr, gamma_ptr, betta_ptr, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> normalize_backward(const int batch,
                                              const int seq_len,
                                              const int hidden_size,
                                              const torch::Tensor& input,
                                              const torch::Tensor& gamma,
                                              const torch::Tensor& betta,
                                              const torch::Tensor& output,
                                              const torch::Tensor& out1_grad,
                                              const torch::Tensor& out2_grad,
                                              torch::Tensor& mean,
                                              torch::Tensor& var,
                                              const bool preln,
                                              const bool wmean,
                                              const float epsilon)
{
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(out1_grad);
    CHECK_INPUT(out2_grad);
    CHECK_INPUT(gamma);
    CHECK_INPUT(betta);
    int bsz_seq = batch * seq_len;

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto gamma_grad = torch::empty({hidden_size}, options);
    auto betta_grad = torch::empty({hidden_size}, options);
    auto input_grad = torch::empty({batch, seq_len, hidden_size}, options);

    const T* input_ptr = (const T*)input.data_ptr();
    const T* out1_grad_ptr = (const T*)out1_grad.data_ptr();
    const T* out2_grad_ptr = (const T*)out2_grad.data_ptr();
    const T* gamma_ptr = (const T*)gamma.data_ptr();
    const T* betta_ptr = (const T*)betta.data_ptr();
    const T* output_ptr = (const T*)output.data_ptr();
    T* gamma_grad_ptr = (T*)gamma_grad.data_ptr();
    T* betta_grad_ptr = (T*)betta_grad.data_ptr();
    T* inp_grad_ptr = (T*)input_grad.data_ptr();
    T* mean_ptr = (T*)mean.data_ptr();
    T* var_ptr = (T*)var.data_ptr();
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    Normalize_Layer<T> _norm(
        typename Normalize_Layer<T>::Config(batch, seq_len, hidden_size, epsilon, true, wmean));
    sycl::queue qs[2] = {q, q};

    _norm.SetMean(mean_ptr);
    _norm.SetVar(var_ptr);

    if (preln) {
        if (wmean)
            _norm.BackwardFusedAdd(bsz_seq,
                                   out1_grad_ptr,
                                   out2_grad_ptr,
                                   gamma_ptr,
                                   gamma_grad_ptr,
                                   betta_grad_ptr,
                                   qs,
                                   inp_grad_ptr,
                                   input_ptr);
        else
            _norm.BackwardFusedAdd(bsz_seq,
                                   out1_grad_ptr,
                                   out2_grad_ptr,
                                   gamma_ptr,
                                   betta_ptr,
                                   gamma_grad_ptr,
                                   betta_grad_ptr,
                                   qs,
                                   inp_grad_ptr,
                                   output_ptr);
    } else {
        if (wmean)
            _norm.Backward(bsz_seq,
                           out1_grad_ptr,
                           gamma_ptr,
                           gamma_grad_ptr,
                           betta_grad_ptr,
                           qs,
                           inp_grad_ptr,
                           input_ptr);
        else {
            _norm.Backward(bsz_seq,
                           out1_grad_ptr,
                           gamma_ptr,
                           betta_ptr,
                           gamma_grad_ptr,
                           betta_grad_ptr,
                           qs,
                           inp_grad_ptr,
                           output_ptr);
        }
    }
    return {input_grad, gamma_grad, betta_grad};
}

template <typename T>
std::vector<torch::Tensor> softmax_forward(int bsz,
                                           int seq_len,
                                           int num_heads,
                                           torch::Tensor& inout,
                                           const torch::Tensor& mask)
{
    CHECK_INPUT(inout);
    CHECK_INPUT(mask);

    T* inout_ptr = (T*)inout.data_ptr();
    const T* mask_ptr = (const T*)mask.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Softmax<T> _softmax = Softmax<T>(typename Softmax<T>::Config(bsz, num_heads, seq_len));
    _softmax.SetSeqLength(seq_len);
    _softmax.Forward(bsz, inout_ptr, mask_ptr, q);
    return {inout};
}

template <typename T>
std::vector<torch::Tensor> softmax_backward(int bsz,
                                            int seq_len,
                                            int num_heads,
                                            torch::Tensor& out_grad,
                                            const torch::Tensor& input)
{
    CHECK_INPUT(out_grad);
    CHECK_INPUT(input);

    T* out_grad_ptr = (T*)out_grad.data_ptr();
    const T* input_ptr = (const T*)input.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();
    Softmax<T> _softmax = Softmax<T>(typename Softmax<T>::Config(bsz, num_heads, seq_len));
    _softmax.SetSeqLength(seq_len);

    _softmax.Backward(bsz, out_grad_ptr, input_ptr, q);
    return {out_grad};
}

template <typename T>
std::vector<torch::Tensor> stridedbatchgemm_forward(const int batchSize,
                                                    const int m,
                                                    const int n,
                                                    const int k,
                                                    const float alpha,
                                                    const float beta,
                                                    const torch::Tensor& matA,
                                                    const torch::Tensor& matB)
{
    CHECK_INPUT(matA);
    CHECK_INPUT(matB);

    auto options = torch::TensorOptions()
                       .dtype(matA.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    StridedBatchGemm<T> _sbgemm =
        StridedBatchGemm<T>(typename StridedBatchGemm<T>::Config(batchSize,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 1.0,
                                                                //  alpha,
                                                                 beta,
                                                                 oneapi::mkl::transpose::trans,
                                                                 oneapi::mkl::transpose::nontrans));

    const T* matA_ptr = (const T*)matA.data_ptr();
    const T* matB_ptr = (const T*)matB.data_ptr();

    auto matC = torch::empty({batchSize, n, m}, options);

    T* matC_ptr = (T*)matC.data_ptr();

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    _sbgemm.Forward(batchSize, matC_ptr, matA_ptr, matB_ptr, q);
    matC *= alpha;
    return {matC};
}

template <typename T>
std::vector<torch::Tensor> stridedbatchgemm_backward(const int batchSize,
                                                     const int m,
                                                     const int n,
                                                     const int k,
                                                     const float alpha,
                                                     const float beta,
                                                     const torch::Tensor& grad_matC,
                                                     const torch::Tensor& matA,
                                                     const torch::Tensor& matB)
{
    CHECK_INPUT(grad_matC);
    CHECK_INPUT(matA);
    CHECK_INPUT(matB);

    auto options = torch::TensorOptions()
                       .dtype(matA.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    StridedBatchGemm<T> _sbgemm =
        StridedBatchGemm<T>(typename StridedBatchGemm<T>::Config(batchSize,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 1.0,
                                                                //  alpha,
                                                                 beta,
                                                                 oneapi::mkl::transpose::trans,
                                                                 oneapi::mkl::transpose::nontrans));

    const T* grad_c_ptr = (const T*)grad_matC.data_ptr();
    const T* matA_ptr = (const T*)matA.data_ptr();
    const T* matB_ptr = (const T*)matB.data_ptr();

    auto grad_matA = torch::empty(matA.sizes(), options);
    auto grad_matB = torch::empty(matB.sizes(), options);
    CHECK_INPUT(grad_matA);
    CHECK_INPUT(grad_matB);

    T* grad_a_ptr = (T*)grad_matA.data_ptr();
    T* grad_b_ptr = (T*)grad_matB.data_ptr();
    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    _sbgemm.Backward(batchSize, grad_c_ptr, matA_ptr, matB_ptr, q, grad_a_ptr, grad_b_ptr);
    grad_matA *= alpha;
    grad_matB *= alpha;
    return {grad_matA, grad_matB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#define DEF_OPS(_name, _dtype)                                                    \
    m.def("dropout_forward_" #_name, &dropout_forward<_dtype>,                    \
          "DROPOUT forward with " #_name " (DPCPP)");                             \
    m.def("dropout_forward_" #_name, &dropout_forward_with_bias<_dtype>,          \
          "DROPOUT forward with bias with " #_name " (DPCPP)");                   \
    m.def("dropout_backward_" #_name, &dropout_backward<_dtype>,                  \
          "DROPOUT backward with " #_name " (DPCPP)");                            \
    m.def("feedforward_forward_" #_name, &feedforward_forward<_dtype>,            \
          "FEEDFORWARD forward with " #_name " (DPCPP)");                         \
    m.def("feedforward_backward_" #_name, &feedforward_backward<_dtype>,          \
          "FEEDFORWARD backward with " #_name " (DPCPP)");                        \
    m.def("fused_add2_" #_name, &fused_add2<_dtype>,                              \
          "Fused add2 with " #_name " (DPCPP)");                                  \
    m.def("transform_0213_" #_name, &transform_0213<_dtype>,                      \
          "transform 0213 with " #_name " (DPCPP)");                              \
    m.def("bias_add_transform_0213_" #_name, &bias_add_transform_0213<_dtype>,    \
          "bias add transform 0213 with " #_name " (DPCPP)");                     \
    m.def("transform4d_0213_" #_name, &transform4d_0213<_dtype>,                  \
          "transform4d 0213 with " #_name " (DPCPP)");                            \
    m.def("gelu_forward_" #_name, &gelu_forward<_dtype>,                          \
          "GELU forward with " #_name " (DPCPP)");                                \
    m.def("gelu_backward_" #_name, &gelu_backward<_dtype>,                        \
          "GELU backward with " #_name " (DPCPP)");                               \
    m.def("normalize_forward_" #_name, &normalize_forward<_dtype>,                \
          "NORMALIZE forward with " #_name " (DPCPP)");                           \
    m.def("normalize_backward_" #_name, &normalize_backward<_dtype>,              \
          "NORMALIZE backward with " #_name " (DPCPP)");                          \
    m.def("softmax_forward_" #_name, &softmax_forward<_dtype>,                    \
          "SOFTMAX forward with " #_name " (DPCPP)");                             \
    m.def("softmax_backward_" #_name, &softmax_backward<_dtype>,                  \
          "SOFTMAX backward with " #_name " (DPCPP)");                            \
    m.def("stridebatchgemm_forward_" #_name, &stridedbatchgemm_forward<_dtype>,   \
          "STRIDEDBATCHGEMM forward with " #_name " (DPCPP)");                    \
    m.def("stridebatchgemm_backward_" #_name, &stridedbatchgemm_backward<_dtype>, \
          "STRIDEDBATCHGEMM backward with " #_name " (DPCPP)")
    DEF_OPS(fp32, float);
    DEF_OPS(fp16, half);
    DEF_OPS(bf16, bf16);
}