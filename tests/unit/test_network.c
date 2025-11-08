/*
 * Unit tests for network.c
 * Tests network creation, layer management, forward pass, training, and utilities
 */

#include "basednn/network.h"
#include "basednn/layer.h"
#include "basednn/optimizer.h"
#include "basednn/tensor.h"
#include "basednn/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...\n", #name); \
    test_##name(); \
    printf("  ✓ test_%s passed\n", #name); \
} while(0)

// ============ Network Creation Tests ============

TEST(network_create_basic) {
    Network *net = network_create();
    
    assert(net != NULL);
    assert(net->layers != NULL);
    assert(net->num_layers == 0);
    assert(net->num_parameters == 0);
    assert(net->capacity == 8);  // Initial capacity
    
    network_free(net);
}

TEST(network_add_single_layer) {
    Network *net = network_create();
    
    LayerConfig config = LINEAR(10, 5);
    Layer *layer = layer_create(config);
    
    network_add_layer(net, layer);
    
    assert(net->num_layers == 1);
    assert(net->layers[0] == layer);
    assert(net->num_parameters > 0);
    
    network_free(net);
}

TEST(network_add_multiple_layers) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(10, 5));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(5, 2));
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    assert(net->num_layers == 3);
    assert(net->layers[0] == layer1);
    assert(net->layers[1] == layer2);
    assert(net->layers[2] == layer3);
    
    network_free(net);
}

TEST(network_capacity_expansion) {
    Network *net = network_create();
    
    // Add more layers than initial capacity
    for (int i = 0; i < 10; i++) {
        Layer *layer = layer_create(RELU());
        network_add_layer(net, layer);
    }
    
    assert(net->num_layers == 10);
    assert(net->capacity >= 10);
    
    network_free(net);
}

// ============ Forward Pass Tests ============

TEST(network_forward_single_layer) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 3));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.5f);
    network_add_layer(net, layer);
    
    size_t input_shape[] = {4};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    // [1, 1, 1, 1] @ weights (all 1) + bias (0.5) = [4.5, 4.5, 4.5]
    for (size_t i = 0; i < output->size; i++) {
        ASSERT_FLOAT_EQ(output->data[i], 4.5f);
    }
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_multi_layer) {
    Network *net = network_create();
    
    // Create 3-layer network: Linear(5->4) -> ReLU -> Linear(4->3)
    Layer *layer1 = layer_create(LINEAR(5, 4));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(4, 3));
    
    tensor_fill(layer1->weights, 0.5f);
    tensor_fill(layer1->bias, 0.0f);
    tensor_fill(layer3->weights, 0.5f);
    tensor_fill(layer3->bias, 0.0f);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    size_t input_shape[] = {5};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_batch) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 2));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    network_add_layer(net, layer);
    
    // Batch of 3 samples
    size_t input_shape[] = {3, 4};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->ndim == 2);
    assert(output->shape[0] == 3);
    assert(output->shape[1] == 2);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_with_activations) {
    Network *net = network_create();
    
    Layer *linear = layer_create(LINEAR(3, 3));
    Layer *sigmoid = layer_create(SIGMOID());
    
    tensor_fill(linear->weights, 0.5f);
    tensor_fill(linear->bias, 0.0f);
    
    network_add_layer(net, linear);
    network_add_layer(net, sigmoid);
    
    size_t input_shape[] = {3};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    // Output should be sigmoid activated (between 0 and 1)
    for (size_t i = 0; i < output->size; i++) {
        assert(output->data[i] > 0.0f && output->data[i] < 1.0f);
    }
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

// ============ Parameter Management Tests ============

TEST(network_get_parameters_empty) {
    Network *net = network_create();
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(num_params == 0);
    assert(params == NULL);
    
    network_free(net);
}

TEST(network_get_parameters_single_layer) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(net, layer);
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(params != NULL);
    assert(num_params == 2);  // Weights and bias
    assert(params[0] == layer->weights);
    assert(params[1] == layer->bias);
    
    network_free(net);
}

TEST(network_get_parameters_multiple_layers) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(5, 4));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(4, 3));
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(params != NULL);
    assert(num_params == 4);  // 2 from layer1 + 0 from layer2 + 2 from layer3
    
    network_free(net);
}

TEST(network_zero_grad) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(3, 2));
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    
    // Allocate and fill gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 2.0f;
    }
    
    network_add_layer(net, layer);
    
    network_zero_grad(net);
    
    // All gradients should be zero
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    network_free(net);
}

// ============ Accuracy Function Tests ============

TEST(network_accuracy_perfect) {
    size_t shape[] = {3, 2};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: class 0
    predictions->data[0] = 0.9f; predictions->data[1] = 0.1f;
    targets->data[0] = 1.0f; targets->data[1] = 0.0f;
    
    // Sample 1: class 1
    predictions->data[2] = 0.2f; predictions->data[3] = 0.8f;
    targets->data[2] = 0.0f; targets->data[3] = 1.0f;
    
    // Sample 2: class 0
    predictions->data[4] = 0.7f; predictions->data[5] = 0.3f;
    targets->data[4] = 1.0f; targets->data[5] = 0.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 1.0f);  // 100% accuracy
    
    tensor_free(predictions);
    tensor_free(targets);
}

TEST(network_accuracy_partial) {
    size_t shape[] = {4, 3};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: predicted class 0, target class 0 (correct)
    predictions->data[0] = 0.8f; predictions->data[1] = 0.1f; predictions->data[2] = 0.1f;
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; targets->data[2] = 0.0f;
    
    // Sample 1: predicted class 1, target class 2 (wrong)
    predictions->data[3] = 0.1f; predictions->data[4] = 0.7f; predictions->data[5] = 0.2f;
    targets->data[3] = 0.0f; targets->data[4] = 0.0f; targets->data[5] = 1.0f;
    
    // Sample 2: predicted class 2, target class 2 (correct)
    predictions->data[6] = 0.1f; predictions->data[7] = 0.2f; predictions->data[8] = 0.7f;
    targets->data[6] = 0.0f; targets->data[7] = 0.0f; targets->data[8] = 1.0f;
    
    // Sample 3: predicted class 0, target class 0 (correct)
    predictions->data[9] = 0.9f; predictions->data[10] = 0.05f; predictions->data[11] = 0.05f;
    targets->data[9] = 1.0f; targets->data[10] = 0.0f; targets->data[11] = 0.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 0.75f);  // 75% accuracy (3 out of 4 correct)
    
    tensor_free(predictions);
    tensor_free(targets);
}

TEST(network_accuracy_zero) {
    size_t shape[] = {2, 2};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: predicted class 0, target class 1 (wrong)
    predictions->data[0] = 0.9f; predictions->data[1] = 0.1f;
    targets->data[0] = 0.0f; targets->data[1] = 1.0f;
    
    // Sample 1: predicted class 0, target class 1 (wrong)
    predictions->data[2] = 0.8f; predictions->data[3] = 0.2f;
    targets->data[2] = 0.0f; targets->data[3] = 1.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 0.0f);  // 0% accuracy
    
    tensor_free(predictions);
    tensor_free(targets);
}

// ============ Edge Cases and Error Handling ============

TEST(network_forward_null_input) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(net, layer);
    
    Tensor *output = network_forward(net, NULL);
    assert(output == NULL);
    
    network_free(net);
}

TEST(network_forward_null_network) {
    size_t shape[] = {3};
    Tensor *input = tensor_ones(shape, 1);
    
    Tensor *output = network_forward(NULL, input);
    assert(output == NULL);
    
    tensor_free(input);
}

TEST(network_forward_empty_network) {
    Network *net = network_create();
    
    size_t shape[] = {3};
    Tensor *input = tensor_ones(shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    // Should return input if no layers
    assert(output == input);
    
    tensor_free(input);
    network_free(net);
}

TEST(network_free_null) {
    network_free(NULL);  // Should not crash
}

TEST(network_add_layer_null_network) {
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(NULL, layer);  // Should not crash
    layer_free(layer);
}

TEST(network_add_layer_null_layer) {
    Network *net = network_create();
    network_add_layer(net, NULL);  // Should not crash
    assert(net->num_layers == 0);
    network_free(net);
}

TEST(network_zero_grad_null) {
    network_zero_grad(NULL);  // Should not crash
}

TEST(network_accuracy_null_inputs) {
    size_t shape[] = {2, 2};
    Tensor *t = tensor_ones(shape, 2);
    
    assert(network_accuracy(NULL, t) == 0.0f);
    assert(network_accuracy(t, NULL) == 0.0f);
    assert(network_accuracy(NULL, NULL) == 0.0f);
    
    tensor_free(t);
}

TEST(network_accuracy_mismatched_shapes) {
    size_t shape_a[] = {3, 2};
    size_t shape_b[] = {2, 2};
    
    Tensor *a = tensor_ones(shape_a, 2);
    Tensor *b = tensor_ones(shape_b, 2);
    
    assert(network_accuracy(a, b) == 0.0f);
    
    tensor_free(a);
    tensor_free(b);
}

// ============ Complex Network Tests ============

TEST(network_deep_network) {
    Network *net = network_create();
    
    // Create a deep network with 5 layers
    network_add_layer(net, layer_create(LINEAR(10, 8)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(8, 6)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(6, 4)));
    
    assert(net->num_layers == 5);
    
    size_t input_shape[] = {10};
    Tensor *input = tensor_randn(input_shape, 1, 42);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 4);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_with_all_activation_types) {
    Network *net = network_create();
    
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(SIGMOID()));
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(TANH()));
    network_add_layer(net, layer_create(LINEAR(5, 3)));
    network_add_layer(net, layer_create(SOFTMAX()));
    
    size_t input_shape[] = {5};
    Tensor *input = tensor_randn(input_shape, 1, 42);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    // Softmax output should sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < output->size; i++) {
        sum += output->data[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_batch_processing) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 2));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    network_add_layer(net, layer);
    
    // Process batches of different sizes
    size_t batch_sizes[] = {1, 5, 10, 32};
    
    for (size_t i = 0; i < 4; i++) {
        size_t input_shape[] = {batch_sizes[i], 4};
        Tensor *input = tensor_ones(input_shape, 2);
        
        Tensor *output = network_forward(net, input);
        
        assert(output != NULL);
        assert(output->shape[0] == batch_sizes[i]);
        assert(output->shape[1] == 2);
        
        tensor_free(input);
        tensor_free(output);
    }
    
    network_free(net);
}

int main() {
    printf("\n=== Running Network Unit Tests ===\n\n");
    
    // Network creation
    RUN_TEST(network_create_basic);
    RUN_TEST(network_add_single_layer);
    RUN_TEST(network_add_multiple_layers);
    RUN_TEST(network_capacity_expansion);
    
    // Forward pass
    RUN_TEST(network_forward_single_layer);
    RUN_TEST(network_forward_multi_layer);
    RUN_TEST(network_forward_batch);
    RUN_TEST(network_forward_with_activations);
    
    // Parameter management
    RUN_TEST(network_get_parameters_empty);
    RUN_TEST(network_get_parameters_single_layer);
    RUN_TEST(network_get_parameters_multiple_layers);
    RUN_TEST(network_zero_grad);
    
    // Accuracy
    RUN_TEST(network_accuracy_perfect);
    RUN_TEST(network_accuracy_partial);
    RUN_TEST(network_accuracy_zero);
    
    // Edge cases
    RUN_TEST(network_forward_null_input);
    RUN_TEST(network_forward_null_network);
    RUN_TEST(network_forward_empty_network);
    RUN_TEST(network_free_null);
    RUN_TEST(network_add_layer_null_network);
    RUN_TEST(network_add_layer_null_layer);
    RUN_TEST(network_zero_grad_null);
    RUN_TEST(network_accuracy_null_inputs);
    RUN_TEST(network_accuracy_mismatched_shapes);
    
    // Complex networks
    RUN_TEST(network_deep_network);
    RUN_TEST(network_with_all_activation_types);
    RUN_TEST(network_batch_processing);
    
    printf("\n=== All Network Tests Passed! ===\n\n");
    return 0;
}
