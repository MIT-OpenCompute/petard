#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Tensor functions
Tensor* tensor_add(Tensor *A, Tensor *B);
Tensor* tensor_sub(Tensor *A, Tensor *B);
Tensor* tensor_mul(Tensor *A, Tensor *B);
Tensor* tensor_matmul(Tensor *A, Tensor *B);
Tensor* tensor_transpose(Tensor *A);

// Activation functions
Tensor* tensor_relu(Tensor *A);
Tensor* tensor_sigmoid(Tensor *A);
Tensor* tensor_tanh(Tensor *A);
Tensor* tensor_softmax(Tensor *A);

// Loss functions
Tensor* tensor_mse(Tensor *predictions, Tensor *targets);
Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets);
Tensor* tensor_binary_cross_entropy(Tensor *predictions, Tensor *targets);

#endif
