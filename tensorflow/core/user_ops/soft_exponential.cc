#include <cmath>
// #include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

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

template <typename T>
class SoftExponentialOp : public OpKernel {
public:
  explicit SoftExponentialOp(OpKernelConstruction *context): OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &alpha_tensor = context->input(0), &features_tensor = context->input(1);
    auto alpha = alpha_tensor.flat<T>(), features = features_tensor.flat<T>();

    Tensor *activations_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, features_tensor.shape(), &activations_tensor));

    auto activations = activations_tensor->flat<float>();

    for (auto i = 0; i < features.size(); ++i) {
      activations(i) = SoftExponential(alpha(i), features(i));
      // std::cout << "SoftExponential" << i << std::endl;
    }
  }

private:
  T SoftExponential(T a, T x) {
    if (a < T(0)) {
      return -log(T(1) - a * (x + a)) / a;
    } else if (a == T(0)) {
      return x;
    } else {
      return (exp(a * x) - T(1)) / a + a;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialOp<double>);

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

template <typename T>
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

    auto alpha_backprops = alpha_backprops_tensor->flat<float>(), backprops = backprops_tensor->flat<float>();

    for (auto i = 0; i < features.size(); ++i) {
      alpha_backprops(i) = SoftExponentialGradAlpha(alpha(i), features(i)) * gradients(i);
      backprops(i) = SoftExponentialGradFeatures(alpha(i), features(i)) * gradients(i);
      // std::cout << "SoftExponentialGrad" << i << std::endl;
    }
  }

private:
  T SoftExponentialGradAlpha(T a, T x) {
    if (a < T(0)) {
      return (log(T(1) - (a * a + a * x)) - (T(2) * a * a + a * x) / (a * a + a * x - T(1))) / a / a;
    } else if (a == T(0)) {
      return x * x / T(2) + T(1);
    } else {
      return T(1) + ((a * x - T(1)) * exp(a * x) + T(1)) / a / a;
    }
  }

  T SoftExponentialGradFeatures(T a, T x) {
    if (a < T(0)) {
      return T(1) / (T(1) - a * (a + x));
    } else {
      return exp(a * x);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponentialGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialGradOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("SoftExponentialGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialGradOp<double>);
