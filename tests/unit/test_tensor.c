#include "petard/tensor.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_tensor_create() {
    printf("Test: tensor_create\n");
    printf("  Creating tensor with shape [2, 3]...\n");
    
    size_t shape[] = {2, 3};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->size == 6);
    assert(t->data != NULL);
    
    printf("  Result: ndim=%zu, shape=[%zu, %zu], size=%zu\n", 
           t->ndim, t->shape[0], t->shape[1], t->size);
    
    tensor_free(t);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_zeroes() {
    printf("Test: tensor_zeroes\n");
    printf("  Creating zero tensor with shape [2, 2]...\n");
    
    size_t shape[] = {2, 2};
    Tensor *t = tensor_zeroes(shape, 2);
    
    assert(t != NULL);
    for (size_t i = 0; i < t->size; i++) {
        assert(t->data[i] == 0.0f);
    }
    
    printf("  Result: all %zu elements are 0.0\n", t->size);
    
    tensor_free(t);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_ones() {
    printf("Test: tensor_ones\n");
    printf("  Creating ones tensor with shape [3, 2]...\n");
    
    size_t shape[] = {3, 2};
    Tensor *t = tensor_ones(shape, 2);
    
    assert(t != NULL);
    for (size_t i = 0; i < t->size; i++) {
        assert(t->data[i] == 1.0f);
    }
    
    printf("  Result: all %zu elements are 1.0\n", t->size);
    
    tensor_free(t);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_randn() {
    printf("Test: tensor_randn\n");
    printf("  Creating random normal tensor with shape [100]...\n");
    
    size_t shape[] = {100};
    Tensor *t = tensor_randn(shape, 1, 42);
    
    assert(t != NULL);
    
    // Check values are distributed (not all the same)
    int different_values = 0;
    for (size_t i = 1; i < t->size; i++) {
        if (fabs(t->data[i] - t->data[0]) > 0.01f) {
            different_values = 1;
            break;
        }
    }
    assert(different_values);
    
    printf("  Result: values are distributed (first 5: %.3f, %.3f, %.3f, %.3f, %.3f)\n",
           t->data[0], t->data[1], t->data[2], t->data[3], t->data[4]);
    
    tensor_free(t);
    printf("  ✓ PASSED\n\n");
}

void test_tensor_requires_grad() {
    printf("Test: tensor_set_requires_grad\n");
    printf("  Creating tensor and setting requires_grad...\n");
    
    size_t shape[] = {2, 2};
    Tensor *t = tensor_ones(shape, 2);
    
    assert(t->requires_grad == 0);
    assert(t->grad == NULL);
    printf("  Initial: requires_grad=%d, grad=%s\n", t->requires_grad, t->grad ? "allocated" : "NULL");
    
    tensor_set_requires_grad(t, 1);
    assert(t->requires_grad == 1);
    printf("  After set: requires_grad=%d\n", t->requires_grad);
    
    tensor_free(t);
    printf("  ✓ PASSED\n\n");
}

int main() {
    printf("Running Tensor Tests\n");

    test_tensor_create();
    test_tensor_zeroes();
    test_tensor_ones();
    test_tensor_randn();
    test_tensor_requires_grad();
    
    return 0;
}
