#include "../include/registry.h"
#include "../include/layer.h"
#include "../include/optimizer.h"
#include "../include/ops.h"
#include <stdlib.h>
#include <string.h>

// ====================================================
// Register Helpers
// ====================================================

#define REGISTRY_SIZE 64

typedef struct RegistryEntry {
    char *key;
    void *value;
    struct RegistryEntry *next;
} RegistryEntry;

typedef struct {
    RegistryEntry *buckets[REGISTRY_SIZE];
} Registry;

static unsigned int hash(const char *str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % REGISTRY_SIZE;
}

static void registry_set(Registry *reg, const char *key, void *value) {
    unsigned int idx = hash(key);
    RegistryEntry *entry = reg->buckets[idx];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    entry = malloc(sizeof(RegistryEntry));
    entry->key = strdup(key);
    entry->value = value;
    entry->next = reg->buckets[idx];
    reg->buckets[idx] = entry;
}

static void* registry_get(Registry *reg, const char *key) {
    unsigned int idx = hash(key);
    RegistryEntry *entry = reg->buckets[idx];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

static void registry_free(Registry *reg) {
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = reg->buckets[i];
        while (entry) {
            RegistryEntry *next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
        reg->buckets[i] = NULL;
    }
}

// ====================================================
// Layer Registers
// ====================================================

typedef struct {
    LayerCreateFn create_fn;
    LayerForwardFn forward_fn;
} LayerRegistryEntry;

static Registry layer_registry = {{NULL}};

void register_layer(const char *name, LayerCreateFn create_fn, LayerForwardFn forward_fn) {
    LayerRegistryEntry *entry = malloc(sizeof(LayerRegistryEntry));
    entry->create_fn = create_fn;
    entry->forward_fn = forward_fn;
    registry_set(&layer_registry, name, entry);
}

LayerCreateFn get_layer_create_fn(const char *name) {
    LayerRegistryEntry *entry = registry_get(&layer_registry, name);
    return entry ? entry->create_fn : NULL;
}

LayerForwardFn get_layer_forward_fn(const char *name) {
    LayerRegistryEntry *entry = registry_get(&layer_registry, name);
    return entry ? entry->forward_fn : NULL;
}

// ====================================================
// Operation Registers
// ====================================================

typedef struct {
    OpFn op_fn;
    int priority;
} OperationRegistryEntry;

static Registry operation_registry = {{NULL}};

void register_operation(const char *name, OpFn op_fn) {
    register_operation_backend(name, op_fn, 0);
}

void register_operation_backend(const char *name, OpFn op_fn, int priority) {
    OperationRegistryEntry *existing = (OperationRegistryEntry*)registry_get(&operation_registry, name);
    
    if (!existing || priority > existing->priority) {
        OperationRegistryEntry *entry = malloc(sizeof(OperationRegistryEntry));
        entry->op_fn = op_fn;
        entry->priority = priority;
        registry_set(&operation_registry, name, entry);
    }
}

OpFn get_operation_fn(const char *name) {
    OperationRegistryEntry *entry = (OperationRegistryEntry*)registry_get(&operation_registry, name);
    return entry ? entry->op_fn : NULL;
}

// ====================================================
// Tensor Operation Registers
// ====================================================

static Registry tensor_op_registry = {{NULL}};

void register_tensor_op(const char *name, BackwardFn backward_fn) {
    registry_set(&tensor_op_registry, name, (void*)backward_fn);
}

BackwardFn get_tensor_op_backward_fn(const char *name) {
    return (BackwardFn)registry_get(&tensor_op_registry, name);
}

// ====================================================
// Optimizer Registers
// ====================================================

typedef struct {
    OptimizerInitStateFn init_state_fn;
    OptimizerStepFn step_fn;
    OptimizerFreeStateFn free_state_fn;
} OptimizerRegistryEntry;

static Registry optimizer_registry = {{NULL}};

void register_optimizer(const char *name,
                       OptimizerInitStateFn init_state_fn,
                       OptimizerStepFn step_fn,
                       OptimizerFreeStateFn free_state_fn) {
    OptimizerRegistryEntry *entry = malloc(sizeof(OptimizerRegistryEntry));
    entry->init_state_fn = init_state_fn;
    entry->step_fn = step_fn;
    entry->free_state_fn = free_state_fn;
    registry_set(&optimizer_registry, name, entry);
}

OptimizerInitStateFn get_optimizer_init_state_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->init_state_fn : NULL;
}

OptimizerStepFn get_optimizer_step_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->step_fn : NULL;
}

OptimizerFreeStateFn get_optimizer_free_state_fn(const char *name) {
    OptimizerRegistryEntry *entry = registry_get(&optimizer_registry, name);
    return entry ? entry->free_state_fn : NULL;
}

// ====================================================
// Registry Initialization
// ====================================================

// Backend initialization
#ifdef HAS_WGPU
extern int wgpu_init(void);
extern void wgpu_register_ops(void);
#endif

static void backend_init_all(void) {
#ifdef HAS_WGPU
    if (wgpu_init() == 0) {
        wgpu_register_ops();
    }
#endif
}

void registry_init() {
    layer_register_builtins();
    ops_register_builtins();
    optimizer_register_builtins();
    backend_init_all();
}

void registry_cleanup() {
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = layer_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&layer_registry);
    
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = operation_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&operation_registry);
    
    registry_free(&tensor_op_registry);
    
    for (int i = 0; i < REGISTRY_SIZE; i++) {
        RegistryEntry *entry = optimizer_registry.buckets[i];
        while (entry) {
            free(entry->value);
            entry = entry->next;
        }
    }
    registry_free(&optimizer_registry);
    
#ifdef HAS_WGPU
    extern void wgpu_cleanup(void);
    wgpu_cleanup();
#endif
}
