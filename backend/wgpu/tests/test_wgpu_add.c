#include "../../../core/include/tensor.h"
#include "../../../core/include/ops.h"
#include "../../../core/include/registry.h"
#include "../wgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Test utilities
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "✗ FAIL: %s\n", msg); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        fprintf(stderr, "✗ FAIL: %s (%.6f != %.6f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("✓ PASS: %s\n", name); \
    tests_passed++; \
} while(0)

// Test cases
void test_add_small_vectors(void) {
    // Test: 1D vectors of size 4
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    
    Tensor *A = tensor_create((size_t[]){4}, 1);
    Tensor *B = tensor_create((size_t[]){4}, 1);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    ASSERT_TRUE(C->size == 4, "Output size mismatch");
    
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected[i], 1e-5f, "Element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_small_vectors");
}

void test_add_large_vectors(void) {
    // Test: Large 1D vectors (>65535 elements to test 2D dispatch)
    size_t size = 100000;
    
    Tensor *A = tensor_create((size_t[]){size}, 1);
    Tensor *B = tensor_create((size_t[]){size}, 1);
    ASSERT_TRUE(A && B, "Failed to create large tensors");
    
    // Initialize with simple pattern
    for (size_t i = 0; i < size; i++) {
        A->data[i] = (float)i;
        B->data[i] = (float)i * 2.0f;
    }
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    ASSERT_TRUE(C->size == size, "Output size mismatch");
    
    // Verify results
    for (size_t i = 0; i < size; i++) {
        float expected = (float)i + (float)i * 2.0f;
        ASSERT_FLOAT_EQ(C->data[i], expected, 1e-3f, "Element mismatch in large vector");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_large_vectors");
}

void test_add_2d_matrices(void) {
    // Test: 2D matrices
    float data_a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float data_b[] = {
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };
    float expected[] = {
        11.0f, 22.0f, 33.0f,
        44.0f, 55.0f, 66.0f
    };
    
    Tensor *A = tensor_create((size_t[]){2, 3}, 2);
    Tensor *B = tensor_create((size_t[]){2, 3}, 2);
    ASSERT_TRUE(A && B, "Failed to create 2D tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    ASSERT_TRUE(C->size == 6, "Output size mismatch");
    
    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected[i], 1e-5f, "Element mismatch in 2D");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_2d_matrices");
}

void test_add_negative_values(void) {
    // Test: Negative values
    float data_a[] = {-1.0f, -2.0f, -3.0f, 4.0f};
    float data_b[] = {5.0f, -6.0f, 7.0f, -8.0f};
    float expected[] = {4.0f, -8.0f, 4.0f, -4.0f};
    
    Tensor *A = tensor_create((size_t[]){4}, 1);
    Tensor *B = tensor_create((size_t[]){4}, 1);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected[i], 1e-5f, "Negative value mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_negative_values");
}

void test_add_zeros(void) {
    // Test: Adding zeros
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    Tensor *A = tensor_create((size_t[]){4}, 1);
    Tensor *B = tensor_create((size_t[]){4}, 1);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(C->data[i], data_a[i], 1e-5f, "Zero addition mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_zeros");
}

void test_add_square_matrix(void) {
    // Test: 256×256 square matrix (common size)
    size_t N = 256;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create square matrices");
    
    // Initialize
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 1.0f;
        B->data[i] = 2.0f;
    }
    
    Tensor *C = tensor_add(A, B);
    ASSERT_TRUE(C != NULL, "tensor_add returned NULL");
    
    // Verify all elements are 3.0
    for (size_t i = 0; i < N * N; i++) {
        ASSERT_FLOAT_EQ(C->data[i], 3.0f, 1e-5f, "Square matrix element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("add_square_matrix");
}

int main(void) {
    printf("=== GPU Tensor Add Unit Tests ===\n\n");
    
    // Initialize registry
    registry_init();
    
    // Initialize GPU backend
    if (wgpu_init() != 0) {
        fprintf(stderr, "Failed to initialize GPU backend\n");
        return 1;
    }
    
    if (!wgpu_available()) {
        fprintf(stderr, "GPU backend not available\n");
        wgpu_cleanup();
        return 1;
    }
    
    // Register GPU operations
    wgpu_register_ops();
    
    // Run tests
    test_add_small_vectors();
    test_add_large_vectors();
    test_add_2d_matrices();
    test_add_negative_values();
    test_add_zeros();
    test_add_square_matrix();
    
    // Cleanup
    wgpu_cleanup();
    registry_cleanup();
    
    // Summary
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
