#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "ops.h"

typedef enum {
    LAYER_LINEAR,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_TANH,
    LAYER_SOFTMAX
} LayerType; 

typedef struct Layer Layer;

struct Layer {
    LayerType type;
    Tensor *weights;
    Tensor *bias;
    Tensor *output;
    Tensor **parameters;
    size_t num_parameters;
    Tensor* (*forward)(Layer *self, Tensor *input);
    void (*free)(Layer *self);
}; 

typedef struct {
    Layer base;
    size_t in_features;
    size_t out_features;
} LinearLayer; 

// Layer constructors/destructor
Layer* layer_linear_create(size_t in_features, size_t out_features);
Layer* layer_relu_create();
Layer* layer_sigmoid_create();
Layer* layer_tanh_create();
Layer* layer_softmax_create();
void layer_free(Layer *layer); 


// Layer operations
Tensor* layer_forward(Layer *layer, Tensor *input);

// Utilities
void layer_zero_grad(Layer *layer);
Tensor** layer_get_parameters(Layer *layer, size_t *num_params);

#endif