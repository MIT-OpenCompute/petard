#include "../../../core/include/tensor.h"
#include "../../../core/include/ops.h"
#include "../../../core/include/registry.h"
#include "../wgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

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

#define TEST_PASS(name) do { \
    printf("✓ PASS: %s\n", name); \
    tests_passed++; \
} while(0)

// Timing utilities
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// CPU-only matmul implementation (bypasses registry)
static Tensor* cpu_only_matmul(Tensor *A, Tensor *B) {
    if (!A || !B || A->ndim != 2 || B->ndim != 2) return NULL;
    
    size_t M = A->shape[0];
    size_t K = A->shape[1];
    size_t N = B->shape[1];
    
    if (B->shape[0] != K) return NULL;
    
    size_t C_shape[2] = {M, N};
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL;
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A->data[i * K + k] * B->data[k * N + j];
            }
            C->data[i * N + j] = acc;
        }
    }
    
    return C;
}

// Verify results match
static int results_match(Tensor *A, Tensor *B, float tolerance) {
    if (!A || !B || A->size != B->size) return 0;
    
    for (size_t i = 0; i < A->size; i++) {
        if (fabsf(A->data[i] - B->data[i]) > tolerance) {
            return 0;
        }
    }
    return 1;
}

void test_matmul_256_gpu_faster(void) {
    // Test: 256×256 matmul should be faster on GPU
    size_t N = 256;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    // Initialize with simple values
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = (float)(i % 100) / 100.0f;
        B->data[i] = (float)((i * 7) % 100) / 100.0f;
    }
    
    // GPU path (via registry)
    double gpu_start = get_time_ms();
    Tensor *C_gpu = tensor_matmul(A, B);
    double gpu_time = get_time_ms() - gpu_start;
    
    ASSERT_TRUE(C_gpu != NULL, "GPU matmul failed");
    
    // CPU path (direct)
    double cpu_start = get_time_ms();
    Tensor *C_cpu = cpu_only_matmul(A, B);
    double cpu_time = get_time_ms() - cpu_start;
    
    ASSERT_TRUE(C_cpu != NULL, "CPU matmul failed");
    
    // Verify results match
    ASSERT_TRUE(results_match(C_gpu, C_cpu, 0.1f), "GPU and CPU results don't match");
    
    // GPU should be significantly faster (at least 2x for 256×256)
    float speedup = cpu_time / gpu_time;
    
    printf("    256×256 matmul: GPU %.2f ms, CPU %.2f ms (%.1fx speedup)\n", 
           gpu_time, cpu_time, speedup);
    
    ASSERT_TRUE(speedup > 2.0f, "GPU not significantly faster than CPU");
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C_gpu);
    tensor_free(C_cpu);
    TEST_PASS("matmul_256_gpu_faster");
}

void test_matmul_512_gpu_faster(void) {
    // Test: 512×512 matmul should show even better GPU speedup
    size_t N = 512;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    // Initialize
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 0.1f;
        B->data[i] = 0.2f;
    }
    
    // GPU path
    double gpu_start = get_time_ms();
    Tensor *C_gpu = tensor_matmul(A, B);
    double gpu_time = get_time_ms() - gpu_start;
    
    ASSERT_TRUE(C_gpu != NULL, "GPU matmul failed");
    
    // CPU path
    double cpu_start = get_time_ms();
    Tensor *C_cpu = cpu_only_matmul(A, B);
    double cpu_time = get_time_ms() - cpu_start;
    
    ASSERT_TRUE(C_cpu != NULL, "CPU matmul failed");
    
    // Verify correctness
    ASSERT_TRUE(results_match(C_gpu, C_cpu, 0.5f), "GPU and CPU results don't match");
    
    // GPU should be much faster (at least 10x for 512×512)
    float speedup = cpu_time / gpu_time;
    
    printf("    512×512 matmul: GPU %.2f ms, CPU %.2f ms (%.1fx speedup)\n", 
           gpu_time, cpu_time, speedup);
    
    ASSERT_TRUE(speedup > 10.0f, "GPU not significantly faster than CPU");
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C_gpu);
    tensor_free(C_cpu);
    TEST_PASS("matmul_512_gpu_faster");
}

void test_matmul_1024_gpu_faster(void) {
    // Test: 1024×1024 matmul should show massive GPU advantage
    size_t N = 1024;
    
    Tensor *A = tensor_create((size_t[]){N, N}, 2);
    Tensor *B = tensor_create((size_t[]){N, N}, 2);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    // Initialize
    for (size_t i = 0; i < N * N; i++) {
        A->data[i] = 0.01f;
        B->data[i] = 0.02f;
    }
    
    // GPU path
    double gpu_start = get_time_ms();
    Tensor *C_gpu = tensor_matmul(A, B);
    double gpu_time = get_time_ms() - gpu_start;
    
    ASSERT_TRUE(C_gpu != NULL, "GPU matmul failed");
    
    // CPU path (only sample for verification since full 1024×1024 CPU is slow)
    double cpu_start = get_time_ms();
    Tensor *C_cpu = cpu_only_matmul(A, B);
    double cpu_time = get_time_ms() - cpu_start;
    
    ASSERT_TRUE(C_cpu != NULL, "CPU matmul failed");
    
    // Verify correctness (sample check)
    int matches = 1;
    for (size_t i = 0; i < N * N; i += 1000) {
        if (fabsf(C_gpu->data[i] - C_cpu->data[i]) > 1.0f) {
            matches = 0;
            break;
        }
    }
    ASSERT_TRUE(matches, "GPU and CPU results don't match");
    
    // GPU should be massively faster (at least 50x for 1024×1024)
    float speedup = cpu_time / gpu_time;
    
    printf("    1024×1024 matmul: GPU %.2f ms, CPU %.2f ms (%.1fx speedup)\n", 
           gpu_time, cpu_time, speedup);
    
    ASSERT_TRUE(speedup > 50.0f, "GPU not significantly faster than CPU");
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C_gpu);
    tensor_free(C_cpu);
    TEST_PASS("matmul_1024_gpu_faster");
}

void test_add_correctness_only(void) {
    // Test: Element-wise add correctness (performance not expected to be better)
    size_t N = 10000;
    
    Tensor *A = tensor_create((size_t[]){N}, 1);
    Tensor *B = tensor_create((size_t[]){N}, 1);
    ASSERT_TRUE(A && B, "Failed to create tensors");
    
    for (size_t i = 0; i < N; i++) {
        A->data[i] = (float)i;
        B->data[i] = (float)i * 2.0f;
    }
    
    // GPU path
    Tensor *C_gpu = tensor_add(A, B);
    ASSERT_TRUE(C_gpu != NULL, "GPU add failed");
    
    // Verify correctness
    for (size_t i = 0; i < N; i++) {
        float expected = (float)i + (float)i * 2.0f;
        ASSERT_TRUE(fabsf(C_gpu->data[i] - expected) < 1e-3f, "Add result incorrect");
    }
    
    printf("    Element-wise add: Correctness verified (bandwidth-bound, speedup not expected)\n");
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C_gpu);
    TEST_PASS("add_correctness_only");
}

int main(void) {
    printf("=== GPU Performance Verification Tests ===\n");
    printf("These tests verify GPU is being used by comparing performance.\n");
    printf("Matmul is compute-intensive and should show significant GPU speedup.\n\n");
    
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
    
    printf("Running performance comparison tests...\n\n");
    
    // Run tests
    test_add_correctness_only();
    test_matmul_256_gpu_faster();
    test_matmul_512_gpu_faster();
    test_matmul_1024_gpu_faster();
    
    // Cleanup
    wgpu_cleanup();
    registry_cleanup();
    
    // Summary
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_passed > 0 && tests_failed == 0) {
        printf("\n✓ GPU is being used and shows expected performance gains!\n");
    }
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
