#include "trebuchet/ops.h"
#include "trebuchet/autograd.h"
#include <stdlib.h>
#include <math.h>

// Helper functions for gradient updates
static void grad_update_two_vars(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op = op_type;
        C->num_inputs = 2;
        C->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->inputs[1] = B;
        C->backward_fn = backward_fn;
    }
}

static void grad_update_one_var(Tensor *A, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad) {
        C->requires_grad = 1;
        C->op = op_type;
        C->num_inputs = 1;
        C->inputs = (Tensor **)malloc(1 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->backward_fn = backward_fn;
    }
}


// Tensor functions
static void tensor_ewise(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (!A || !B || !C) return; 
    if (A->ndim != B->ndim || A->ndim != C->ndim) return; 
    for (size_t i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) return;
    }

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = func(A->data[i], B->data[i]); 
    }

    grad_update_two_vars(A, B, C, func, op_type, backward_fn);
}

static float add_func(float x, float y) { return x + y; }
static float sub_func(float x, float y) { return x - y; }
static float mul_func(float x, float y) { return x * y; }

Tensor* tensor_add(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, add_func, OP_ADD, backward_add);
    return C;
}

Tensor* tensor_sub(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, sub_func, OP_SUB, backward_sub);
    return C;
}

Tensor* tensor_mul(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, mul_func, OP_MUL, backward_mul);
    return C;
}

Tensor* tensor_matmul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL; 
    if (A->ndim != 2 || B->ndim != 2) return NULL; 
    if (A->shape[1] != B->shape[0]) return NULL; 

    size_t C_shape[2] = {A->shape[0], B->shape[1]}; 
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL; 

    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < B->shape[1]; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < A->shape[1]; k++) {
                acc += A->data[i * A->shape[1] + k] * B->data[k * B->shape[1] + j];
            }
            C->data[i * C->shape[1] + j] = acc; 
        }
    }

    grad_update_two_vars(A, B, C, NULL, OP_MATMUL, backward_matmul);

    return C; 
}

Tensor* tensor_transpose(Tensor *A) {
    if (!A) return NULL; 
    if (A->ndim != 2) return NULL; 

    size_t C_shape[2] = {A->shape[1], A->shape[0]}; 
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL;

    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < A->shape[1]; j++) {
            C->data[j * C->shape[1] + i] = A->data[i * A->shape[1] + j]; 
        }
    }

    grad_update_one_var(A, C, NULL, OP_TRANSPOSE, backward_transpose);

    return C; 
}

// Activation functions
Tensor* tensor_relu(Tensor *A) {
    if (!A) return NULL;

    Tensor *C = tensor_create(A->shape, A->ndim); 
    if (!C) return NULL; 

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = A->data[i] > 0.0f ? A->data[i] : 0.0f; 
    }

    grad_update_one_var(A, C, NULL, OP_RELU, backward_relu);

    return C; 
}

Tensor* tensor_sigmoid(Tensor *A) {
    if (!A) return NULL; 

    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = 1.0f / (1.0f + expf(-A->data[i])); 
    }

    grad_update_one_var(A, C, NULL, OP_SIGMOID, backward_sigmoid);

    return C;
}

Tensor* tensor_tanh(Tensor *A) {
    if (!A) return NULL; 

    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = tanhf(A->data[i]); 
    }

    grad_update_one_var(A, C, NULL, OP_TANH, backward_tanh);

    return C;
}

Tensor* tensor_softmax(Tensor *A) {
    if (!A) return NULL; 
    
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL; 

    float max_val = A->data[0]; 
    for (size_t i = 1; i < A->size; i++) {
        if (A->data[i] > max_val) max_val = A->data[i];
    }

    float sum = 0.0f; 
    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = expf(A->data[i] - max_val);
        sum += C->data[i];
    }

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] /= sum; 
    }

    grad_update_one_var(A, C, NULL, OP_SOFTMAX, backward_softmax);

    return C;
}

// Loss functions
Tensor* tensor_mse(Tensor *predictions, Tensor *targets) {
    if (!predictions || !targets) return NULL; 
    if (predictions->ndim != targets->ndim) return NULL; 
    for (size_t i = 0; i < predictions->ndim; i++) {
        if (predictions->shape[i] != targets->shape[i]) return NULL; 
    }

    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL; 

    float sum_sq_erro = 0.0f; 
    for (size_t i = 0; i < predictions->size; i++) {
        float diff = predictions->data[i] - targets->data[i];
        sum_sq_erro += diff * diff;
    }
    loss->data[0] = sum_sq_erro / predictions->size;
    return loss;
}

Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets) {
    
}