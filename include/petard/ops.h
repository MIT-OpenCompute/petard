#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Tensor operations
Tensor* tensor_add(Tensor *A, Tensor *B);
Tensor* tensor_sub(Tensor *A, Tensor *B);
Tensor* tensor_mul(Tensor *A, Tensor *B);
Tensor* tensor_matmul(Tensor *A, Tensor *B);
Tensor* tensor_transpose(Tensor *A);

// Activation operations
Tensor* tensor_relu(Tensor *A);
Tensor* tensor_sigmoid(Tensor *A);
Tensor* tensor_tanh(Tensor *A);
Tensor* tensor_softmax(Tensor *A);

#endif
