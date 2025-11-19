#ifndef WGPU_BACKEND_H
#define WGPU_BACKEND_H

#include "../../core/include/tensor.h"

// Initialize WGPU backend (finds adapter, creates device and queue)
// Returns 0 on success, -1 on failure
int wgpu_init(void);

// Cleanup WGPU resources
void wgpu_cleanup(void);

// Check if WGPU backend is available and initialized
int wgpu_available(void);

// Register WGPU operations with the registry
void wgpu_register_ops(void);

// Cleanup operation-specific caches (called by wgpu_cleanup)
void wgpu_cleanup_pipeline_caches(void);

// Internal helpers (exposed for wgpu_ops.c)
#include <webgpu.h>
#include <wgpu.h>

WGPUBuffer wgpu_create_buffer(uint64_t size, WGPUBufferUsageFlags usage);
void wgpu_write_buffer(WGPUBuffer buffer, uint64_t offset, const void *data, uint64_t size);
int wgpu_read_buffer(WGPUBuffer buffer, uint64_t offset, void *dest, uint64_t size);
WGPUDevice wgpu_get_device(void);
WGPUQueue wgpu_get_queue(void);

#endif // WGPU_BACKEND_H
