#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

// Tensor function gradients
void backward_add(Tensor *out);
void backward_sub(Tensor *out);
void backward_mul(Tensor *out);
void backward_matmul(Tensor *out);
void backward_transpose(Tensor *out);

// Activation function gradients
void backward_relu(Tensor *out);
void backward_sigmoid(Tensor *out);
void backward_tanh(Tensor *out);
void backward_softmax(Tensor *out);

// Loss function gradients
void backward_mse(Tensor *out);
void backward_cross_entropy(Tensor *out);
void backward_binary_cross_entropy(Tensor *out);

#endif
