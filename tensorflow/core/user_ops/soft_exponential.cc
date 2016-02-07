//#include <cmath>
// #include <iostream>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("SoftExponential")
    .Input("alpha: T")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes soft exponential.

See [A continuum among logarithmic, linear, and exponential functions, and its potential to improve generalization in
neural networks](http://arxiv.org/abs/1602.01321 "Luke B. Godfrey, Michael S. Gashler")
)doc");

template <typename Device, typename T>
class SoftExponentialOp : public OpKernel {
public:
  explicit SoftExponentialOp(OpKernelConstruction *context): OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &alpha_tensor = context->input(0), &features_tensor = context->input(1);
    auto alpha = alpha_tensor.flat<T>(), features = features_tensor.flat<T>();

    Tensor *activations_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, features_tensor.shape(), &activations_tensor));

    auto activations = activations_tensor->flat<T>();

    auto device = context->eigen_device<Device>();

    auto one = alpha.constant(T(1));
    auto logarithmic = -(one - alpha * (features + alpha)).log() / alpha;
    auto exponential = ((alpha * features).exp() - one) / alpha + alpha;

    activations.device(device) = (alpha < T(0)).select(
        logarithmic,
        (alpha == T(0)).select(
            features,
            exponential));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialOp<CPUDevice, double>);

// #if GOOGLE_CUDA
//
// REGISTER_KERNEL_BUILDER(
//     Name("SoftExponential").Device(DEVICE_GPU).TypeConstraint<float>("T"), SoftExponentialOp<GPUDevice, float>);
// REGISTER_KERNEL_BUILDER(
//     Name("SoftExponential").Device(DEVICE_GPU).TypeConstraint<double>("T"), SoftExponentialOp<GPUDevice, double>);
//
// #endif

REGISTER_OP("SoftExponentialGrad")
    .Input("gradients: T")
    .Input("alpha: T")
    .Input("features: T")
    .Output("alpha_backprops: T")
    .Output("backprops: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients for the soft exponential operation.

gradients: The backpropagated gradients to the corresponding SoftExponential operation.
alpha: The alpha parameter.
features: The features of the corresponding SoftExponential operation.
alpha_backprops:
backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
`gradients` otherwise.
)doc");

template <typename Device, typename T>
class SoftExponentialGradOp : public OpKernel {
public:
  explicit SoftExponentialGradOp(OpKernelConstruction *context): OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &gradients_tensor =
        context->input(0), &alpha_tensor = context->input(1), &features_tensor = context->input(2);
    auto gradients = gradients_tensor.flat<T>(), alpha = alpha_tensor.flat<T>(), features = features_tensor.flat<T>();

    Tensor *alpha_backprops_tensor = nullptr, *backprops_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, features_tensor.shape(), &alpha_backprops_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, features_tensor.shape(), &backprops_tensor));

    auto alpha_backprops = alpha_backprops_tensor->flat<T>(), backprops = backprops_tensor->flat<T>();

    auto device = context->eigen_device<Device>();

    auto logarithmic = alpha < T(0), linear = alpha == T(0);
    auto one = alpha.constant(T(1)), two = alpha.constant(T(2));

    auto alpha_alpha = alpha * alpha, alpha_features = alpha * features;

    auto logarithmic_grad_alpha = ((one - (alpha_alpha + alpha_features)).log() - (
        two * alpha_alpha + alpha_features) / (alpha_alpha + alpha_features - one)) / alpha_alpha;
    auto linear_grad_alpha = features * features / two + one;
    auto exponential_grad_alpha = one + ((alpha_features - one) * (alpha_features).exp() + one) / alpha_alpha;

    auto logarithmic_grad_features = one / (one - alpha_alpha - alpha_features);
    auto exponential_grad_features = (alpha_features).exp();

    alpha_backprops.device(device) = gradients * logarithmic.select(
        logarithmic_grad_alpha,
        linear.select(
            linear_grad_alpha,
            exponential_grad_alpha));

    backprops.device(device) = gradients * logarithmic.select(
        logarithmic_grad_features,
        exponential_grad_features);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponentialGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SoftExponentialGrad")
    .Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialGradOp<CPUDevice, double>);

// #if GOOGLE_CUDA
//
// REGISTER_KERNEL_BUILDER(
//     Name("SoftExponentialGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), SoftExponentialGradOp<GPUDevice, float>);
// REGISTER_KERNEL_BUILDER(Name("SoftExponentialGrad")
//     .Device(DEVICE_GPU).TypeConstraint<double>("T"), SoftExponentialGradOp<GPUDevice, double>);
//
// #endif
