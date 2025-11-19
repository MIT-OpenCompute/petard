// Heavy workload add shader - performs computation multiple times to increase GPU load

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // global_invocation_id gives the global thread coordinate across all workgroups
    // Linear index calculation for 2D workgroup grid
    let threads_per_row = 65535u * 256u;  // max workgroups in X * threads per workgroup
    let index = global_id.y * threads_per_row + global_id.x;
    
    // Bounds check
    if (index < arrayLength(&output)) {
        // Perform computation multiple times to increase GPU load
        var result: f32 = input_a[index] + input_b[index];
        
        // Do some extra work to make GPU usage visible
        for (var i: i32 = 0; i < 1000; i++) {
            result = result * 1.0 + 0.0;  // No-op but keeps GPU busy
        }
        
        output[index] = result;
    }
}
