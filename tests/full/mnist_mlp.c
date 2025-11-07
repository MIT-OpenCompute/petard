#include "../../include/trebuchet/trebuchet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple MNIST-like data generator (for demonstration)
// In practice, you'd load actual MNIST data
void generate_sample_data(Tensor *X, Tensor *y, size_t batch_size) {
    // Generate random input (28*28 = 784 features)
    for (size_t i = 0; i < X->size; i++) {
        X->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Generate random labels (one-hot encoded, 10 classes)
    for (size_t i = 0; i < y->size; i++) {
        y->data[i] = 0.0f;
    }
    for (size_t i = 0; i < batch_size; i++) {
        int label = rand() % 10;
        y->data[i * 10 + label] = 1.0f;
    }
}

// Calculate accuracy
float loss_accuracy(Tensor *predictions, Tensor *targets, size_t batch_size) {
    int correct = 0;
    for (size_t i = 0; i < batch_size; i++) {
        int pred_class = 0;
        float max_pred = predictions->data[i * 10];
        for (int j = 1; j < 10; j++) {
            if (predictions->data[i * 10 + j] > max_pred) {
                max_pred = predictions->data[i * 10 + j];
                pred_class = j;
            }
        }
        
        int true_class = 0;
        for (int j = 0; j < 10; j++) {
            if (targets->data[i * 10 + j] == 1.0f) {
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    return (float)correct / batch_size;
}

int main() {
    srand(time(NULL));
    
    // Hyperparameters
    const size_t input_size = 784;    // 28x28 MNIST images
    const size_t hidden_size = 128;
    const size_t output_size = 10;    // 10 classes (digits 0-9)
    const size_t batch_size = 32;
    const size_t num_epochs = 10;
    const float learning_rate = 0.01f;
    
    printf("=== MNIST MLP Classifier ===\n");
    printf("Architecture: %zu -> %zu -> %zu\n", input_size, hidden_size, output_size);
    printf("Batch size: %zu, Epochs: %zu, LR: %.4f\n\n", batch_size, num_epochs, learning_rate);
    
    // Build MLP network
    Network *net = network_create();
    network_add_layer(net, layer_linear_create(input_size, hidden_size));
    network_add_layer(net, layer_relu_create());
    network_add_layer(net, layer_linear_create(hidden_size, output_size));
    network_add_layer(net, layer_softmax_create());
    
    printf("Network created with %zu layers\n\n", net->num_layers);
    
    // Create optimizer
    Optimizer *optimizer = optimizer_sgd_from_network(net, learning_rate, 0.9f);
    
    // Training data tensors
    size_t input_shape[2] = {batch_size, input_size};
    size_t target_shape[2] = {batch_size, output_size};
    
    Tensor *X_train = tensor_create(input_shape, 2);
    Tensor *y_train = tensor_create(target_shape, 2);
    
    // Training loop
    printf("Training:\n");
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        // Generate training batch
        generate_sample_data(X_train, y_train, batch_size);
        
        // Forward pass
        Tensor *output = network_forward(net, X_train);
        
        // Compute loss
        Tensor *loss = loss_cross_entropy(output, y_train);
        float loss_value = loss->data[0];
        
        // Backward pass
        optimizer_zero_grad(optimizer);
        tensor_backward(loss);
        
        // Update parameters
        optimizer_step(optimizer);
        
        // Calculate accuracy
        float acc = loss_accuracy(output, y_train, batch_size);
        
        printf("Epoch %zu/%zu - Loss: %.4f, Accuracy: %.2f%%\n", 
               epoch + 1, num_epochs, loss_value, acc * 100.0f);
        
        // Cleanup tensors from this iteration
        tensor_free(loss);
    }
    
    printf("Inference\n");
    
    // Generate test batch
    Tensor *X_test = tensor_create(input_shape, 2);
    Tensor *y_test = tensor_create(target_shape, 2);
    generate_sample_data(X_test, y_test, batch_size);
    
    // Forward pass (inference mode - no gradients needed)
    Tensor *test_output = network_forward(net, X_test);
    
    // Calculate test loss and accuracy
    float test_loss = loss_cross_entropy_value(test_output, y_test);
    float test_acc = loss_accuracy(test_output, y_test, batch_size);
    
    printf("Test Loss: %.4f, Test Accuracy: %.2f%%\n", test_loss, test_acc * 100.0f);
    
    // Cleanup
    tensor_free(X_train);
    tensor_free(y_train);
    tensor_free(X_test);
    tensor_free(y_test);
    optimizer_free(optimizer);
    network_free(net);
    
    printf("\n=== Training Complete ===\n");
    
    return 0;
}
