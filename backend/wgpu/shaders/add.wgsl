// Element-wise addition: C[i] = A[i] + B[i]

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // global_invocation_id gives the global thread coordinate across all workgroups
    // With workgroup_size(256), there are 256 threads per workgroup in X dimension
    // For 2D dispatch (X_groups, Y_groups, 1), we get:
    //   global_id.x ranges from 0 to (X_groups * 256 - 1)
    //   global_id.y ranges from 0 to (Y_groups - 1)
    // But wait - with 2D workgroup dispatch, threads are only along X within each workgroup
    // So global_id.x = workgroup_id.x * 256 + local_id.x
    // and global_id.y = workgroup_id.y
    
    // Linear index calculation for 2D workgroup grid
    let threads_per_row = 65535u * 256u;  // max workgroups in X * threads per workgroup
    let index = global_id.y * threads_per_row + global_id.x;
    
    // Bounds check
    if (index < arrayLength(&output)) {
        output[index] = input_a[index] + input_b[index];
    }
}
