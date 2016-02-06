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
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("SoftExponential").Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialOp<double>);

REGISTER_OP("SoftExponentialGrad")
    .Input("gradients: T")
    .Input("alpha: T")
    .Input("outputs: T")
    .Output("alpha_backprops: T")
    .Output("backprops: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients for the soft exponential operation.

gradients: The backpropagated gradients to the corresponding SoftExponential operation.
alpha: The alpha parameter.
outputs: The outputs of the corresponding SoftExponential operation.
alpha_backprops:
backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
`gradients` otherwise.
)doc");

template <typename T>
class SoftExponentialGradOp : public OpKernel {
public:
  explicit SoftExponentialGradOp(OpKernelConstruction *context): OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SoftExponentialGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), SoftExponentialGradOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("SoftExponentialGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), SoftExponentialGradOp<double>);
