#include "common.hpp"
#include "context.hpp"
#include "softmax.hpp"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#define DEF_OPS(_name, _dtype)                           \
    m.def("forward_" #_name, &softmax_forward<_dtype>,   \
          "SOFTMAX forward with " #_name " (DPCPP)");    \
    m.def("backward_" #_name, &softmax_backward<_dtype>, \
          "SOFTMAX backward with " #_name " (DPCPP)")
    DEF_OPS(fp32, float);
    DEF_OPS(fp16, half);
    DEF_OPS(bf16, bf16);
}
