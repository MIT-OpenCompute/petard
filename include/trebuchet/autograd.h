#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

// Tensor function gradients
void backward_add(Tensor *C);
void backward_sub(Tensor *C);
void backward_mul(Tensor *C);
void backward_matmul(Tensor *C);
void backward_transpose(Tensor *C);

// Activation function gradients
void backward_relu(Tensor *C);
void backward_sigmoid(Tensor *C);
void backward_tanh(Tensor *C);
void backward_softmax(Tensor *C);

// Loss function gradients
void backward_mse(Tensor *C);
void backward_cross_entropy(Tensor *C);
void backward_binary_cross_entropy(Tensor *C);

#endif
