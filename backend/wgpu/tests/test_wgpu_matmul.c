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
        fprintf(stderr, "✗ FAIL: %s (%.6f != %.6f, diff=%.6f)\n", msg, (float)(a), (float)(b), fabsf((a)-(b))); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("✓ PASS: %s\n", name); \
    tests_passed++; \
} while(0)

// Test cases
void test_matmul_small_square(void) {
    // Test: 2×2 matrices
    float data_a[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    float data_b[] = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    // Expected: [1*5+2*7, 1*6+2*8]   = [19, 22]
    //           [3*5+4*7, 3*6+4*8]   = [43, 50]
    float expected[] = {
        19.0f, 22.0f,
        43.0f, 50.0f
    };
    
    Tensor *A = tensor_create((size_t[]){2, 2}, 2);
    Tensor *B = tensor_create((size_t[]){2, 2}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    ASSERT_TRUE(C->ndim == 2 && C->shape[0] == 2 && C->shape[1] == 2, "Output shape mismatch");
    
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected[i], 1e-4f, "Element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_small_square");
}

void test_matmul_rectangular(void) {
    // Test: 3×2 × 2×4 = 3×4
    float data_a[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    float data_b[] = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    };
    // Expected: Row 0: [1*1+2*0, 1*0+2*1, 1*1+2*0, 1*0+2*1] = [1, 2, 1, 2]
    //           Row 1: [3*1+4*0, 3*0+4*1, 3*1+4*0, 3*0+4*1] = [3, 4, 3, 4]
    //           Row 2: [5*1+6*0, 5*0+6*1, 5*1+6*0, 5*0+6*1] = [5, 6, 5, 6]
    float expected[] = {
        1.0f, 2.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 5.0f, 6.0f
    };
    
    Tensor *A = tensor_create((size_t[]){3, 2}, 2);
    Tensor *B = tensor_create((size_t[]){2, 4}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    ASSERT_TRUE(C->ndim == 2 && C->shape[0] == 3 && C->shape[1] == 4, "Output shape mismatch");
    
    for (size_t i = 0; i < 12; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected[i], 1e-4f, "Element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_rectangular");
}

void test_matmul_identity(void) {
    // Test: A × I = A (identity matrix)
    float data_a[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float data_i[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    
    Tensor *A = tensor_create((size_t[]){3, 3}, 2);
    Tensor *I = tensor_create((size_t[]){3, 3}, 2);
    ASSERT_TRUE(A && I, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(I->data, data_i, sizeof(data_i));
    
    Tensor *C = tensor_matmul(A, I);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    
    // Result should equal A
    for (size_t i = 0; i < 9; i++) {
        ASSERT_FLOAT_EQ(C->data[i], data_a[i], 1e-4f, "Identity multiplication failed");
    }
    
    tensor_free(A);
    tensor_free(I);
    tensor_free(C);
    TEST_PASS("matmul_identity");
}

void test_matmul_128x128(void) {
    // Test: 128×128 matrices (exercises tile boundaries)
    size_t N = 128;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create 128×128 tensors");
    
    // Initialize: A with row index, B with column index
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A->data[i * N + j] = (float)i;
            B->data[i * N + j] = (float)j;
        }
    }
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    ASSERT_TRUE(C->shape[0] == N && C->shape[1] == N, "Output shape mismatch");
    
    // Verify: C[i,j] = sum_k(A[i,k] * B[k,j]) = sum_k(i * j) = i * j * N
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            float expected = (float)i * (float)j * (float)N;
            ASSERT_FLOAT_EQ(C->data[i * N + j], expected, 1e-2f, "128×128 element mismatch");
        }
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_128x128");
}

void test_matmul_256x256(void) {
    // Test: 256×256 matrices (exact tile alignment)
    size_t N = 256;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create 256×256 tensors");
    
    // Simple initialization for verification
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 1.0f;
        B->data[i] = 2.0f;
    }
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    
    // Expected: each element should be 1*2*N = 2*N = 512
    float expected = 2.0f * (float)N;
    for (size_t i = 0; i < N * N; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected, 1.0f, "256×256 element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_256x256");
}

void test_matmul_non_tile_aligned(void) {
    // Test: 100×100 matrices (not tile-aligned, tests padding)
    size_t N = 100;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create 100×100 tensors");
    
    // Initialize with simple values
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 0.5f;
        B->data[i] = 2.0f;
    }
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    
    // Expected: 0.5 * 2.0 * 100 = 100.0
    float expected = 0.5f * 2.0f * (float)N;
    for (size_t i = 0; i < N * N; i++) {
        ASSERT_FLOAT_EQ(C->data[i], expected, 1.0f, "Non-aligned element mismatch");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_non_tile_aligned");
}

void test_matmul_large_512x512(void) {
    // Test: 512×512 matrices (stress test)
    size_t N = 512;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create 512×512 tensors");
    
    // Initialize
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 0.1f;
        B->data[i] = 0.2f;
    }
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    
    // Expected: 0.1 * 0.2 * 512 = 10.24
    float expected = 0.1f * 0.2f * (float)N;
    
    // Check a sampling of elements (checking all 262k takes too long)
    for (size_t i = 0; i < N; i += 17) {  // Sample every 17th row
        for (size_t j = 0; j < N; j += 19) {  // Sample every 19th column
            ASSERT_FLOAT_EQ(C->data[i * N + j], expected, 0.5f, "512×512 sampled element mismatch");
        }
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_large_512x512");
}

void test_matmul_zeros(void) {
    // Test: Multiplication with zeros
    float data_a[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    float data_b[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };
    
    Tensor *A = tensor_create((size_t[]){2, 2}, 2);
    Tensor *B = tensor_create((size_t[]){2, 2}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    memcpy(A->data, data_a, sizeof(data_a));
    memcpy(B->data, data_b, sizeof(data_b));
    
    Tensor *C = tensor_matmul(A, B);
    ASSERT_TRUE(C != NULL, "tensor_matmul returned NULL");
    
    // All elements should be 0
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(C->data[i], 0.0f, 1e-5f, "Zero multiplication failed");
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    TEST_PASS("matmul_zeros");
}

int main(void) {
    printf("=== GPU Tensor Matmul Unit Tests ===\n\n");
    
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
    test_matmul_small_square();
    test_matmul_rectangular();
    test_matmul_identity();
    test_matmul_128x128();
    test_matmul_256x256();
    test_matmul_non_tile_aligned();
    test_matmul_large_512x512();
    test_matmul_zeros();
    
    // Cleanup
    wgpu_cleanup();
    registry_cleanup();
    
    // Summary
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
