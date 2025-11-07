#include "trebuchet/tensor.h"
#include "trebuchet/ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_tensor_add() {
    printf("Test: tensor_add\n");
    printf("  Computing [1,1,1,1] + [1,1,1,1]...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    Tensor *c = tensor_add(a, b);
    
    assert(c != NULL);
    for (size_t i = 0; i < c->size; i++) {
        assert(fabs(c->data[i] - 2.0f) < 1e-6);
    }
    
    printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n", 
           c->data[0], c->data[1], c->data[2], c->data[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_sub() {
    printf("Test: tensor_sub\n");
    printf("  Computing [1,1,1,1] - [1,1,1,1]...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    Tensor *c = tensor_sub(a, b);
    
    assert(c != NULL);
    for (size_t i = 0; i < c->size; i++) {
        assert(fabs(c->data[i] - 0.0f) < 1e-6);
    }
    
    printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n", 
           c->data[0], c->data[1], c->data[2], c->data[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_mul() {
    printf("Test: tensor_mul (element-wise)\n");
    printf("  Computing [1,1,1,1] * [2,3,1,1]...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    b->data[0] = 2.0f;
    b->data[1] = 3.0f;
    
    Tensor *c = tensor_mul(a, b);
    
    assert(c != NULL);
    assert(fabs(c->data[0] - 2.0f) < 1e-6);
    assert(fabs(c->data[1] - 3.0f) < 1e-6);
    
    printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n", 
           c->data[0], c->data[1], c->data[2], c->data[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_matmul() {
    printf("Test: tensor_matmul (matrix multiplication)\n");
    printf("  Computing [[1,1],[1,1]] @ [[1,1],[1,1]]...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    Tensor *c = tensor_matmul(a, b);
    
    assert(c != NULL);
    assert(c->shape[0] == 2);
    assert(c->shape[1] == 2);
    // [[1,1],[1,1]] @ [[1,1],[1,1]] = [[2,2],[2,2]]
    for (size_t i = 0; i < c->size; i++) {
        assert(fabs(c->data[i] - 2.0f) < 1e-6);
    }
    
    printf("  Result: [[%.1f, %.1f], [%.1f, %.1f]]\n", 
           c->data[0], c->data[1], c->data[2], c->data[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_transpose() {
    printf("Test: tensor_transpose\n");
    printf("  Transposing 2x3 matrix [0,1,2,3,4,5]...\n");
    
    size_t shape[] = {2, 3};
    Tensor *a = tensor_create(shape, 2);
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = (float)i;
    }
    
    Tensor *b = tensor_transpose(a);
    
    assert(b != NULL);
    assert(b->shape[0] == 3);
    assert(b->shape[1] == 2);
    assert(fabs(b->data[0] - 0.0f) < 1e-6); // a[0,0]
    assert(fabs(b->data[1] - 3.0f) < 1e-6); // a[1,0]
    assert(fabs(b->data[2] - 1.0f) < 1e-6); // a[0,1]
    
    printf("  Result shape: [%zu, %zu]\n", b->shape[0], b->shape[1]);
    printf("  Result: [[%.1f, %.1f], [%.1f, %.1f], [%.1f, %.1f]]\n",
           b->data[0], b->data[1], b->data[2], b->data[3], b->data[4], b->data[5]);
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_relu() {
    printf("Test: tensor_relu\n");
    printf("  Applying ReLU to [-1, 0, 1, 2]...\n");
    
    size_t shape[] = {4};
    Tensor *a = tensor_create(shape, 1);
    a->data[0] = -1.0f;
    a->data[1] = 0.0f;
    a->data[2] = 1.0f;
    a->data[3] = 2.0f;
    
    Tensor *b = tensor_relu(a);
    
    assert(b != NULL);
    assert(fabs(b->data[0] - 0.0f) < 1e-6);
    assert(fabs(b->data[1] - 0.0f) < 1e-6);
    assert(fabs(b->data[2] - 1.0f) < 1e-6);
    assert(fabs(b->data[3] - 2.0f) < 1e-6);
    
    printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n", 
           b->data[0], b->data[1], b->data[2], b->data[3]);
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_sigmoid() {
    printf("Test: tensor_sigmoid\n");
    printf("  Applying sigmoid to [0, 0]...\n");
    
    size_t shape[] = {2};
    Tensor *a = tensor_zeroes(shape, 1);
    
    Tensor *b = tensor_sigmoid(a);
    
    assert(b != NULL);
    // sigmoid(0) = 0.5
    for (size_t i = 0; i < b->size; i++) {
        assert(fabs(b->data[i] - 0.5f) < 1e-6);
    }
    
    printf("  Result: [%.3f, %.3f] (expected: 0.5 for sigmoid(0))\n", 
           b->data[0], b->data[1]);
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_tanh() {
    printf("Test: tensor_tanh\n");
    printf("  Applying tanh to [0, 0]...\n");
    
    size_t shape[] = {2};
    Tensor *a = tensor_zeroes(shape, 1);
    
    Tensor *b = tensor_tanh(a);
    
    assert(b != NULL);
    // tanh(0) = 0
    for (size_t i = 0; i < b->size; i++) {
        assert(fabs(b->data[i] - 0.0f) < 1e-6);
    }
    
    printf("  Result: [%.3f, %.3f] (expected: 0.0 for tanh(0))\n", 
           b->data[0], b->data[1]);
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_softmax() {
    printf("Test: tensor_softmax\n");
    printf("  Applying softmax to [1, 1, 1]...\n");
    
    size_t shape[] = {3};
    Tensor *a = tensor_ones(shape, 1);
    
    Tensor *b = tensor_softmax(a);
    
    assert(b != NULL);
    // Equal inputs should give equal probabilities
    float sum = 0.0f;
    for (size_t i = 0; i < b->size; i++) {
        assert(fabs(b->data[i] - 1.0f/3.0f) < 1e-5);
        sum += b->data[i];
    }
    // Sum should be 1
    assert(fabs(sum - 1.0f) < 1e-5);
    
    printf("  Result: [%.3f, %.3f, %.3f] (sum=%.3f)\n", 
           b->data[0], b->data[1], b->data[2], sum);
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

int main() {
    printf("Running Ops Tests\n");
    
    test_tensor_add();
    test_tensor_sub();
    test_tensor_mul();
    test_tensor_matmul();
    test_tensor_transpose();
    test_tensor_relu();
    test_tensor_sigmoid();
    test_tensor_tanh();
    test_tensor_softmax();

    return 0;
}
