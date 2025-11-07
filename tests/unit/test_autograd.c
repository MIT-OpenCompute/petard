#include "trebuchet/tensor.h"
#include "trebuchet/ops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_backward_add() {
    printf("Test: backward_add\n");
    printf("  Computing gradients for c = a + b...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_add(a, b);
    
    // Initialize gradient of output
    tensor_backward(c);
    
    // Gradient of add: da = dc, db = dc
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    for (size_t i = 0; i < a->size; i++) {
        assert(fabs(a->grad[i] - 1.0f) < 1e-6);
        assert(fabs(b->grad[i] - 1.0f) < 1e-6);
    }
    
    printf("  Result: a.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           a->grad[0], a->grad[1], a->grad[2], a->grad[3]);
    printf("          b.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           b->grad[0], b->grad[1], b->grad[2], b->grad[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_backward_sub() {
    printf("Test: backward_sub\n");
    printf("  Computing gradients for c = a - b...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_sub(a, b);
    tensor_backward(c);
    
    // Gradient of sub: da = dc, db = -dc
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    for (size_t i = 0; i < a->size; i++) {
        assert(fabs(a->grad[i] - 1.0f) < 1e-6);
        assert(fabs(b->grad[i] - (-1.0f)) < 1e-6);
    }
    
    printf("  Result: a.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           a->grad[0], a->grad[1], a->grad[2], a->grad[3]);
    printf("          b.grad = [%.1f, %.1f, %.1f, %.1f] (negative for subtraction)\n", 
           b->grad[0], b->grad[1], b->grad[2], b->grad[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_backward_mul() {
    printf("Test: backward_mul\n");
    printf("  Computing gradients for c = a * b (element-wise)...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    a->data[0] = 2.0f;
    b->data[0] = 3.0f;
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_mul(a, b);
    tensor_backward(c);
    
    // Gradient of mul: da = dc * b, db = dc * a
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    assert(fabs(a->grad[0] - 3.0f) < 1e-6); // dc * b[0]
    assert(fabs(b->grad[0] - 2.0f) < 1e-6); // dc * a[0]
    
    printf("  Result: a.grad[0] = %.1f (= dc * b[0] = 1.0 * 3.0)\n", a->grad[0]);
    printf("          b.grad[0] = %.1f (= dc * a[0] = 1.0 * 2.0)\n", b->grad[0]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_backward_relu() {
    printf("Test: backward_relu\n");
    printf("  Computing gradients for ReLU([-1, 0, 1, 2])...\n");
    
    size_t shape[] = {4};
    Tensor *a = tensor_create(shape, 1);
    a->data[0] = -1.0f;
    a->data[1] = 0.0f;
    a->data[2] = 1.0f;
    a->data[3] = 2.0f;
    
    tensor_set_requires_grad(a, 1);
    
    Tensor *b = tensor_relu(a);
    tensor_backward(b);
    
    // Gradient of relu: da = dc if a > 0 else 0
    assert(a->grad != NULL);
    assert(fabs(a->grad[0] - 0.0f) < 1e-6); // a[0] < 0
    assert(fabs(a->grad[1] - 0.0f) < 1e-6); // a[1] = 0
    assert(fabs(a->grad[2] - 1.0f) < 1e-6); // a[2] > 0
    assert(fabs(a->grad[3] - 1.0f) < 1e-6); // a[3] > 0
    
    printf("  Result: a.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           a->grad[0], a->grad[1], a->grad[2], a->grad[3]);
    printf("          (zero for negative inputs, 1 for positive)\n");
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_backward_sigmoid() {
    printf("Test: backward_sigmoid\n");
    printf("  Computing gradients for sigmoid([0, 0])...\n");
    
    size_t shape[] = {2};
    Tensor *a = tensor_zeroes(shape, 1);
    
    tensor_set_requires_grad(a, 1);
    
    Tensor *b = tensor_sigmoid(a);
    tensor_backward(b);
    
    // Gradient of sigmoid at 0: sig(0) * (1 - sig(0)) = 0.5 * 0.5 = 0.25
    assert(a->grad != NULL);
    for (size_t i = 0; i < a->size; i++) {
        assert(fabs(a->grad[i] - 0.25f) < 1e-6);
    }
    
    printf("  Result: a.grad = [%.3f, %.3f]\n", a->grad[0], a->grad[1]);
    printf("          (sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25)\n");
    
    tensor_free(a);
    tensor_free(b);
    printf("  ✓ PASSED\n\n");
}

void test_backward_matmul() {
    printf("Test: backward_matmul\n");
    printf("  Computing gradients for c = a @ b (matrix multiplication)...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    Tensor *b = tensor_ones(shape, 2);
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_matmul(a, b);
    tensor_backward(c);
    
    // Gradients should be non-zero
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    for (size_t i = 0; i < a->size; i++) {
        assert(a->grad[i] > 0.0f);
        assert(b->grad[i] > 0.0f);
    }
    
    printf("  Result: a.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           a->grad[0], a->grad[1], a->grad[2], a->grad[3]);
    printf("          b.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           b->grad[0], b->grad[1], b->grad[2], b->grad[3]);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_backward_chain() {
    printf("Test: backward_chain\n");
    printf("  Computing gradients through chain: a -> ReLU -> b -> add -> c...\n");
    
    size_t shape[] = {2, 2};
    Tensor *a = tensor_ones(shape, 2);
    tensor_set_requires_grad(a, 1);
    
    // Chain: a -> relu -> b -> add -> c
    Tensor *b = tensor_relu(a);
    Tensor *c = tensor_add(b, b);
    
    tensor_backward(c);
    
    // Should propagate through the chain
    assert(a->grad != NULL);
    for (size_t i = 0; i < a->size; i++) {
        assert(a->grad[i] > 0.0f);
    }
    
    printf("  Result: a.grad = [%.1f, %.1f, %.1f, %.1f]\n", 
           a->grad[0], a->grad[1], a->grad[2], a->grad[3]);
    printf("          (gradient propagated through ReLU and addition)\n");
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    printf("  ✓ PASSED\n\n");
}

void test_backward_complex() {
    printf("Test: backward_complex (multi-branch computation graph)\n");
    printf("  Graph: x -> [mul by 2, mul by 3] -> add -> sigmoid -> output\n");
    
    size_t shape[] = {2, 2};
    Tensor *x = tensor_ones(shape, 2);
    x->data[0] = 0.5f;
    x->data[1] = 1.0f;
    x->data[2] = 1.5f;
    x->data[3] = 2.0f;
    
    Tensor *two = tensor_ones(shape, 2);
    two->data[0] = 2.0f;
    two->data[1] = 2.0f;
    two->data[2] = 2.0f;
    two->data[3] = 2.0f;
    
    Tensor *three = tensor_ones(shape, 2);
    three->data[0] = 3.0f;
    three->data[1] = 3.0f;
    three->data[2] = 3.0f;
    three->data[3] = 3.0f;
    
    tensor_set_requires_grad(x, 1);
    
    printf("  Input x: [%.1f, %.1f, %.1f, %.1f]\n", 
           x->data[0], x->data[1], x->data[2], x->data[3]);
    
    // Build computation graph:
    // left = x * 2, right = x * 3
    // sum = left + right = 5x
    // output = sigmoid(sum)
    Tensor *left = tensor_mul(x, two);
    Tensor *right = tensor_mul(x, three);
    Tensor *sum = tensor_add(left, right);
    Tensor *output = tensor_sigmoid(sum);
    
    printf("  Output: [%.3f, %.3f, %.3f, %.3f]\n",
           output->data[0], output->data[1], output->data[2], output->data[3]);
    
    tensor_backward(output);
    
    assert(x->grad != NULL);
    
    // Gradient should be: d(output)/dx = d(sigmoid(5x))/dx = 5 * sigmoid'(5x)
    printf("  Result: x.grad = [%.3f, %.3f, %.3f, %.3f]\n",
           x->grad[0], x->grad[1], x->grad[2], x->grad[3]);
    
    // Verify gradients are reasonable (should be between 0 and 5)
    for (size_t i = 0; i < x->size; i++) {
        assert(x->grad[i] > 0.0f);
        assert(x->grad[i] <= 5.0f);
    }
    
    printf("          (gradient accumulated from two branches: 2x and 3x paths)\n");
    
    tensor_free(x);
    tensor_free(two);
    tensor_free(three);
    tensor_free(left);
    tensor_free(right);
    tensor_free(sum);
    tensor_free(output);
    printf("  ✓ PASSED\n\n");
}

void test_backward_matmul_chain() {
    printf("Test: backward_matmul_chain (neural network-like computation)\n");
    printf("  Computing: output = ReLU(x @ W1 + y @ W2) @ W2\n");
    
    size_t shape_x[] = {1, 2};  // 1x2 input
    size_t shape_y[] = {1, 2};  // 1x2 output
    size_t shape_w1[] = {2, 3}; // 2x3 weight matrix
    size_t shape_w2[] = {3, 2}; // 3x2 weight matrix
    
    Tensor *x = tensor_ones(shape_x, 2);
    x->data[0] = 1.0f;
    x->data[1] = 2.0f;

    Tensor *y = tensor_ones(shape_y, 2);
    y->data[0] = 5.0f;
    y->data[1] = 6.0f;
    
    Tensor *W1 = tensor_randn(shape_w1, 2, 50);
    tensor_print(W1);
    Tensor *W2 = tensor_ones(shape_w2, 2);
    
    tensor_set_requires_grad(x, 1);
    tensor_set_requires_grad(y, 1);
    tensor_set_requires_grad(W1, 1);
    tensor_set_requires_grad(W2, 1);
    
    printf("  Input x: [%.1f, %.1f]\n", x->data[0], x->data[1]);
    printf("  Input y: [%.1f, %.1f]\n", y->data[0], y->data[1]);
    
    // Forward pass: x @ W1 + y -> ReLU -> hidden @ W2 -> output
    Tensor *hidden0 = tensor_matmul(x, W1);     // [1, 3]
    Tensor *hidden1 = tensor_matmul(y, W1);           // [1, 2]
    Tensor *hidden2 = tensor_add(hidden0, hidden1); // [1, 3]
    Tensor *hidden3 = tensor_relu(hidden2);       // [1, 3]
    Tensor *output = tensor_matmul(hidden3, W2);     // [1, 2]

    
    printf("  Output: [%.1f, %.1f]\n", output->data[0], output->data[1]);
    
    tensor_backward(output);
    
    // All gradients should be computed
    assert(x->grad != NULL);
    assert(y->grad != NULL);
    assert(W1->grad != NULL);
    assert(W2->grad != NULL);
    
    printf("  Result: x.grad = [%.1f, %.1f]\n", x->grad[0], x->grad[1]);
    printf("          y.grad = [%.1f, %.1f]\n", y->grad[0], y->grad[1]);
    printf("          W1.grad sum = %.1f (gradient for first layer weights)\n", 
           W1->grad[0] + W1->grad[1] + W1->grad[2] + W1->grad[3] + W1->grad[4] + W1->grad[5]);
    printf("          W2.grad sum = %.1f (gradient for second layer weights)\n",
           W2->grad[0] + W2->grad[1] + W2->grad[2] + W2->grad[3] + W2->grad[4] + W2->grad[5]);
    // Verify non-zero gradients (since ReLU will pass through positive values)
    assert(x->grad[0] != 0.0f || x->grad[1] != 0.0f);
    
    printf("          (gradients propagated through 2-layer network)\n");
    
    tensor_free(x);
    tensor_free(y); 
    tensor_free(W1);
    tensor_free(W2);
    tensor_free(hidden0);
    tensor_free(hidden1);
    tensor_free(hidden2);
    tensor_free(hidden3);
    tensor_free(output);
    printf("  ✓ PASSED\n\n");
}

int main() {
    printf("Running autograd tests...\n\n");
    
    test_backward_add();
    test_backward_sub();
    test_backward_mul();
    test_backward_relu();
    test_backward_sigmoid();
    test_backward_matmul();
    test_backward_chain();
    test_backward_complex();
    test_backward_matmul_chain();
    
    return 0;
}
