/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <utility>

#include "utils/compatible.hpp"

#include "dnnl_backend.hpp"
#include "dnnl_opset.hpp"
#include "kernels/kernels.hpp"
#include "patterns/binary_fusion.hpp"
#include "patterns/bn_fusion.hpp"
#include "patterns/conv_fusion.hpp"
#include "patterns/convtranspose_fusion.hpp"
#include "patterns/eltwise_fusion.hpp"
#include "patterns/gelu_fusion.hpp"
#include "patterns/interpolate_fusion.hpp"
#include "patterns/matmul_fusion.hpp"
#include "patterns/pool_fusion.hpp"
#include "patterns/quantize_fusion.hpp"
#include "patterns/reduction_fusion.hpp"
#include "patterns/reorder_fusion.hpp"
#include "patterns/shuffle_fusion.hpp"
#include "patterns/single_op_pattern.hpp"
#include "patterns/sum_fusion.hpp"
#include "tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

bool dnnl_layout_id_manager_t::is_mem_desc_equal(
        const impl::utils::any_t &mem_desc1,
        const impl::utils::any_t &mem_desc2) const {
    auto &md1 = impl::utils::any_cast<const memory::desc &>(mem_desc1);
    auto &md2 = impl::utils::any_cast<const memory::desc &>(mem_desc2);
    return md1 == md2;
}

dnnl_backend::dnnl_backend(const std::string &name, float priority)
    : backend(name, priority) {
    bool ret = register_op_schemas() && register_passes() && register_kernels();
    if (!ret) { throw std::runtime_error(name + " initialize failed"); }
}

bool dnnl_backend::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

bool dnnl_backend::register_passes() {
    DNNL_BACKEND_REGISTER_PASSES_CALL(binary_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(conv_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(convtranspose_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(gelu_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(matmul_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(single_op_pass, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(pool_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(eltwise_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(quantize_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(interpolate_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(sum_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(reorder_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(shuffle_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(reduction_fusion, pass_registry_);
    pass_registry_.sort_passes();

    return true;
}

bool dnnl_backend::register_kernels() {
#if defined(__GNUC__)
#define DNNL_GRAPH_ATTR_UNUSED __attribute__((unused))
#else
#define DNNL_GRAPH_ATTR_UNUSED
#endif
    // Register DNNL kernel
#define DECLARE_KERNEL_EX(kernel_class_, counter) \
    static DNNL_GRAPH_ATTR_UNUSED auto \
            _registered_dnnl_kernel_##kernel_class_##_##counter##_

#define DECLARE_KERNEL(kernel_class_, counter) \
    DECLARE_KERNEL_EX(kernel_class_, counter)

#define DNNL_REGISTER_KERNEL(op_kind_, kernel_class_) \
    DECLARE_KERNEL(kernel_class_, __COUNTER__) \
            = kernel_registry_.register_kernel(op_kind_, \
                    &kernel_registry_t::create_kernel<kernel_class_>);

    // concat
    DNNL_REGISTER_KERNEL(impl::op_kind::Concat, float_concat);

    // conv related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Convolution, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_sigmoid, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_swish, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_hardtanh, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_square, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_tanh, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_abs, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_sqrt, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_depthwise, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_simple_resblock, float_conv_fwd)
    DNNL_REGISTER_KERNEL(
            op_kind::conv_bias_post_ops_chain_fusion, float_conv_fwd)

    DNNL_REGISTER_KERNEL(impl::op_kind::ConvolutionBackpropData, conv_bwd_data)

    DNNL_REGISTER_KERNEL(impl::op_kind::ConvolutionBackpropFilters,
            convolution_backward_weights)
    DNNL_REGISTER_KERNEL(
            op_kind::conv_bwd_f_biasadd_bwd, convolution_backward_weights)

    // convtranspose related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::ConvTranspose, float_convtranspose_fwd)
    DNNL_REGISTER_KERNEL(op_kind::convtranspose_fusion, float_convtranspose_fwd)
    // bn related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::BatchNormInference, batchnorm_fwd_t)
    DNNL_REGISTER_KERNEL(op_kind::bn_relu, batchnorm_fwd_t)
    DNNL_REGISTER_KERNEL(
            impl::op_kind::BatchNormForwardTraining, batchnorm_fwd_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::BatchNormTrainingBackprop,
            batch_normalization_backward)

    // binary operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Add, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::add_relu, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::add_sigmoid, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::add_multiply, binary_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::Multiply, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::multiply_add, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::multiply_relu, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::multiply_sigmoid, binary_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::Maximum, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::maximum_add, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::maximum_relu, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::maximum_sigmoid, binary_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::Minimum, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::minimum_add, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::minimum_relu, binary_t)
    DNNL_REGISTER_KERNEL(op_kind::minimum_sigmoid, binary_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::Divide, binary_t)

    // bias_add
    DNNL_REGISTER_KERNEL(impl::op_kind::BiasAdd, bias_add)

    // elementwise related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Abs, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Elu, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Exp, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::GELU, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::HardSwish, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::HardTanh, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Log, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Pow, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReLU, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Round, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Sqrt, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Square, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::Tanh, float_eltwise_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReLUBackprop, eltwise_backward)
    DNNL_REGISTER_KERNEL(impl::op_kind::GELUBackprop, eltwise_backward)

    // matmul related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::MatMul, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_elu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_sigmoid, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_hardtanh, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_relu6, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_elu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_sigmoid, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_swish, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_hardtanh, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_add, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_add_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_bn, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_sigmoid, float_matmul)

    // pooling related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::AvgPool, float_pooling_fwd)
    DNNL_REGISTER_KERNEL(op_kind::avgpool_add, float_pooling_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::MaxPool, float_pooling_fwd)
    DNNL_REGISTER_KERNEL(op_kind::maxpool_add, float_pooling_fwd)
    DNNL_REGISTER_KERNEL(impl::op_kind::AvgPoolBackprop, pooling_backward)
    DNNL_REGISTER_KERNEL(impl::op_kind::MaxPoolBackprop, pooling_backward)

    // softmax operators
    DNNL_REGISTER_KERNEL(impl::op_kind::SoftMax, softmax_fwd_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::SoftMaxBackprop, softmax_backward)

    // logsoftmax operators
    DNNL_REGISTER_KERNEL(impl::op_kind::LogSoftmax, logsoftmax_fwd_t)
    DNNL_REGISTER_KERNEL(impl::op_kind::LogSoftmaxBackprop, logsoftmax_backward)

    // layernorm kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::LayerNorm, layernorm_fwd_t)

    //interpolate kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Interpolate, float_resampling_fwd)
    DNNL_REGISTER_KERNEL(op_kind::interpolate_fusion, float_resampling_fwd)
    DNNL_REGISTER_KERNEL(
            impl::op_kind::InterpolateBackprop, resampling_backward)

    // reorder kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Reorder, float_reorder)
    DNNL_REGISTER_KERNEL(impl::op_kind::TypeCast, float_reorder)
    DNNL_REGISTER_KERNEL(op_kind::reorder_sum, float_reorder)
    DNNL_REGISTER_KERNEL(op_kind::int8_reorder, quantized_reorder)

    // prelu kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::PReLU, float_prelu_fwd)

    // reduction operators
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceL1, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceL2, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceMax, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceMean, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceMin, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceProd, float_reduction)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReduceSum, float_reduction)
    DNNL_REGISTER_KERNEL(op_kind::dnnl_reduction, float_reduction)
    DNNL_REGISTER_KERNEL(op_kind::reduction_fusion, float_reduction)

    // quantize and dequantize kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Quantize, quantize_dequantize)
    DNNL_REGISTER_KERNEL(impl::op_kind::Dequantize, quantize_dequantize)

    // quantized conv
    DNNL_REGISTER_KERNEL(op_kind::int8_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias_add, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_bias_add_relu, quantized_conv)

    // quantized convtranspose
    DNNL_REGISTER_KERNEL(op_kind::int8_convtranspose, quantized_convtranspose)
    DNNL_REGISTER_KERNEL(
            op_kind::int8_convtranspose_bias, quantized_convtranspose)

    // quantized matmul
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8x8float_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8x8float_matmul_div, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8x8float_matmul_div_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_MHA, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::f32_MHA, float_matmul);
    DNNL_REGISTER_KERNEL(op_kind::matmul_div, float_matmul);
    DNNL_REGISTER_KERNEL(op_kind::matmul_div_add, float_matmul);

    //eltwise+binary ops
    DNNL_REGISTER_KERNEL(op_kind::relu_add, float_eltwise_fwd);
    DNNL_REGISTER_KERNEL(op_kind::chained_relu, float_eltwise_fwd);

    // quantized pooling
    DNNL_REGISTER_KERNEL(op_kind::int8_maxpool, quantized_pooling);
    DNNL_REGISTER_KERNEL(op_kind::int8_maxpool_add, quantized_pooling);
    DNNL_REGISTER_KERNEL(op_kind::int8_avgpool, quantized_pooling);
    DNNL_REGISTER_KERNEL(op_kind::int8_avgpool_add, quantized_pooling);

    // quantized eltwise
    DNNL_REGISTER_KERNEL(op_kind::int8_relu, quantized_eltwise);
    DNNL_REGISTER_KERNEL(op_kind::int8_relu_add, quantized_eltwise);

    // sum fusion
    DNNL_REGISTER_KERNEL(op_kind::dnnl_sum, sum_t);

    // shuffle fusion
    DNNL_REGISTER_KERNEL(op_kind::dnnl_shuffle, shuffle_fwd_t);

#undef DNNL_REGISTER_KERNEL
#undef DNNL_GRAPH_ATTR_UNUSED

    return true;
}

size_t dnnl_backend::get_mem_size(const impl::logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool dnnl_backend::compare_logical_tensor(const impl::logical_tensor_t &lhs,
        const impl::logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

impl::utils::optional<size_t> dnnl_backend::set_mem_desc(
        const impl::utils::any_t &mem_desc) {
    return layout_id_manager_.set_mem_desc(mem_desc);
}

impl::utils::optional<impl::utils::any_t> dnnl_backend::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

} // namespace dnnl_impl

// This function should be called by backend_registry_t
void register_dnnl_backend() {
    backend_registry_t::get_singleton().register_backend(
            &dnnl_impl::dnnl_backend::get_singleton());
}

} // namespace impl
} // namespace graph
} // namespace dnnl
