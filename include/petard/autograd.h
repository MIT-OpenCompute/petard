#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

// Gradients
void backward_add(Tensor *out);
void backward_sub(Tensor *out);
void backward_mul(Tensor *out);
void backward_matmul(Tensor *out);
void backward_transpose(Tensor *out);
void backward_relu(Tensor *out);
void backward_sigmoid(Tensor *out);
void backward_tanh(Tensor *out);
void backward_softmax(Tensor *out);

#endif
