#include "trebuchet/autograd.h"
#include <stdlib.h>

// Tensor function gradients
void backward_add(Tensor *C) {
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] += C->grad[i];
        }
    }
}

void backward_sub(Tensor *C) {
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] -= C->grad[i];
        }
    }
}

void backward_mul(Tensor *C) {
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i] * B->data[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] += C->grad[i] * A->data[i];
        }
    }
}

void backward_matmul(Tensor *C) {
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t k = 0; k < A->shape[1]; k++) {
                for (size_t j = 0; j < B->shape[1]; j++) {
                    A->grad[i * A->shape[1] + k] += 
                        C->grad[i * B->shape[1] + j] * B->data[k * B->shape[1] + j];
                }
            }
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t k = 0; k < B->shape[0]; k++) {
            for (size_t j = 0; j < B->shape[1]; j++) {
                for (size_t i = 0; i < A->shape[0]; i++) {
                    B->grad[k * B->shape[1] + j] += 
                        C->grad[i * B->shape[1] + j] * A->data[i * A->shape[1] + k];
                }
            }
        }
    }
}

void backward_transpose(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                A->grad[i * A->shape[1] + j] += C->grad[j * A->shape[0] + i];
            }
        }
    }
}

// Activation function gradients
void backward_relu(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i] * (A->data[i] > 0 ? 1.0f : 0.0f);
        }
    }
}

void backward_sigmoid(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            float sig = C->data[i];
            A->grad[i] += C->grad[i] * sig * (1.0f - sig);
        }
    }
}

void backward_tanh(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            float t = C->data[i];
            A->grad[i] += C->grad[i] * (1.0f - t * t);
        }
    }
}

void backward_softmax(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            for (size_t j = 0; j < A->size; j++) {
                float delta = (i == j) ? 1.0f : 0.0f;
                A->grad[i] += C->grad[j] * C->data[j] * (delta - C->data[i]);
            }
        }
    }
}

// Loss function gradients
void backward_mse(Tensor *C) {
    Tensor *predictions = C->inputs[0];
    Tensor *targets = C->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (2.0f / predictions->size) * (predictions->data[i] - targets->data[i]) * C->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                (2.0f / targets->size) * (predictions->data[i] - targets->data[i]) * C->grad[0];
        }
    }
}

void backward_cross_entropy(Tensor *C) {
    Tensor *predictions = C->inputs[0];
    Tensor *targets = C->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (-targets->data[i] / predictions->data[i]) * C->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                ( -logf(predictions->data[i]) ) * C->grad[0];
        }
    }
}

void backward_binary_cross_entropy(Tensor *C) {
    Tensor *predictions = C->inputs[0];
    Tensor *targets = C->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (-(targets->data[i] / predictions->data[i]) + 
                 (1.0f - targets->data[i]) / (1.0f - predictions->data[i])) * C->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                (-logf(predictions->data[i]) + 
                 -logf(1.0f - predictions->data[i])) * C->grad[0];
        }
    }
}