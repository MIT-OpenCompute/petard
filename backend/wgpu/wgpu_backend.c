#include "wgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <webgpu.h>
#include <wgpu.h>

// Global state
static WGPUInstance instance = NULL;
static WGPUAdapter adapter = NULL;
static WGPUDevice device = NULL;
static WGPUQueue queue = NULL;
static int initialized = 0;

// ====================================================
// Callbacks
// ====================================================

static void request_adapter_callback(WGPURequestAdapterStatus status,
                                      WGPUAdapter received_adapter,
                                      char const *message,
                                      void *userdata) {
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter*)userdata = received_adapter;
    } else {
        fprintf(stderr, "[WGPU] Failed to request adapter: %s\n", message ? message : "Unknown error");
    }
}

static void request_device_callback(WGPURequestDeviceStatus status,
                                     WGPUDevice received_device,
                                     char const *message,
                                     void *userdata) {
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice*)userdata = received_device;
    } else {
        fprintf(stderr, "[WGPU] Failed to request device: %s\n", message ? message : "Unknown error");
    }
}

static void device_error_callback(WGPUErrorType type, char const *message, void *userdata) {
    fprintf(stderr, "[WGPU] Device error (%d): %s\n", type, message ? message : "Unknown error");
}

// ====================================================
// Initialization
// ====================================================

int wgpu_init(void) {
    if (initialized) {
        return 0;  // Already initialized
    }

    // Create instance
    WGPUInstanceDescriptor inst_desc = {0};
    instance = wgpuCreateInstance(&inst_desc);
    if (!instance) {
        fprintf(stderr, "[WGPU] Failed to create instance\n");
        return -1;
    }

    // Request adapter (prefer high-performance GPU)
    WGPURequestAdapterOptions adapter_options = {
        .nextInChain = NULL,
        .compatibleSurface = NULL,
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .backendType = WGPUBackendType_Undefined,
        .forceFallbackAdapter = 0,
    };

    adapter = NULL;
    wgpuInstanceRequestAdapter(instance, &adapter_options, request_adapter_callback, &adapter);

    // Poll for adapter
    int max_attempts = 100;
    for (int i = 0; i < max_attempts && !adapter; i++) {
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);
#else
        // Simple sleep alternative - just poll
        for (volatile int j = 0; j < 1000000; j++);
#endif
    }

    if (!adapter) {
        fprintf(stderr, "[WGPU] Failed to obtain adapter\n");
        wgpuInstanceRelease(instance);
        return -1;
    }

    // Verify we're using a GPU (not CPU fallback)
    WGPUAdapterProperties props = {0};
    wgpuAdapterGetProperties(adapter, &props);
    
    if (props.adapterType == WGPUAdapterType_CPU) {
        fprintf(stderr, "[WGPU] WARNING: Using CPU adapter, not GPU!\n");
    }

    // Request device
    WGPUDeviceDescriptor device_desc = {
        .label = "baseDNN Device",
    };

    device = NULL;
    wgpuAdapterRequestDevice(adapter, &device_desc, request_device_callback, &device);

    // Poll for device
    for (int i = 0; i < max_attempts && !device; i++) {
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);
#else
        for (volatile int j = 0; j < 1000000; j++);
#endif
    }

    if (!device) {
        fprintf(stderr, "[WGPU] Failed to obtain device\n");
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        return -1;
    }

    // Set error callback
    wgpuDeviceSetUncapturedErrorCallback(device, device_error_callback, NULL);

    // Get queue
    queue = wgpuDeviceGetQueue(device);
    if (!queue) {
        fprintf(stderr, "[WGPU] Failed to get queue\n");
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        return -1;
    }

    initialized = 1;
    return 0;
}

void wgpu_cleanup(void) {
    if (!initialized) return;

    // Cleanup operation-specific caches first
    wgpu_cleanup_pipeline_caches();

    if (queue) wgpuQueueRelease(queue);
    if (device) wgpuDeviceRelease(device);
    if (adapter) wgpuAdapterRelease(adapter);
    if (instance) wgpuInstanceRelease(instance);

    queue = NULL;
    device = NULL;
    adapter = NULL;
    instance = NULL;
    initialized = 0;
}

int wgpu_available(void) {
    return initialized && device != NULL && queue != NULL;
}

// ====================================================
// Buffer Management Helpers
// ====================================================

WGPUBuffer wgpu_create_buffer(uint64_t size, WGPUBufferUsageFlags usage) {
    if (!wgpu_available()) return NULL;

    WGPUBufferDescriptor buffer_desc = {
        .nextInChain = NULL,
        .label = NULL,
        .usage = usage,
        .size = size,
        .mappedAtCreation = 0,
    };

    return wgpuDeviceCreateBuffer(device, &buffer_desc);
}

void wgpu_write_buffer(WGPUBuffer buffer, uint64_t offset, const void *data, uint64_t size) {
    if (!wgpu_available() || !buffer) return;
    wgpuQueueWriteBuffer(queue, buffer, offset, data, size);
}

// Synchronous buffer read with proper completion wait
typedef struct {
    void *dest;
    size_t size;
    int done;
} BufferReadData;

static void buffer_map_callback(WGPUBufferMapAsyncStatus status, void *userdata) {
    BufferReadData *read_data = (BufferReadData*)userdata;
    
    if (status == WGPUBufferMapAsyncStatus_Success) {
        read_data->done = 1;
    } else {
        fprintf(stderr, "[WGPU] Buffer map failed with status %d\n", status);
        read_data->done = -1;
    }
}

int wgpu_read_buffer(WGPUBuffer buffer, uint64_t offset, void *dest, uint64_t size) {
    if (!wgpu_available() || !buffer || !dest) return -1;

    BufferReadData read_data = {
        .dest = dest,
        .size = size,
        .done = 0,
    };

    // Map buffer for reading
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, offset, size, buffer_map_callback, &read_data);

    // Poll until mapping complete
    int max_attempts = 10000;
    for (int i = 0; i < max_attempts && read_data.done == 0; i++) {
        wgpuDevicePoll(device, 1, NULL);
        
        // Small delay
        for (volatile int j = 0; j < 10000; j++);
    }

    if (read_data.done != 1) {
        fprintf(stderr, "[WGPU] Buffer read timeout\n");
        return -1;
    }

    // Copy data
    const void *mapped_data = wgpuBufferGetConstMappedRange(buffer, offset, size);
    if (mapped_data) {
        memcpy(dest, mapped_data, size);
    } else {
        fprintf(stderr, "[WGPU] Failed to get mapped range\n");
        wgpuBufferUnmap(buffer);
        return -1;
    }

    wgpuBufferUnmap(buffer);
    return 0;
}

WGPUDevice wgpu_get_device(void) {
    return device;
}

WGPUQueue wgpu_get_queue(void) {
    return queue;
}

