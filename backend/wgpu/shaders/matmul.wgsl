// Matrix multiplication: C = A × B
// A: M×K, B: K×N, C: M×N
// Uses 16×16 tiling with workgroup shared memory for high performance

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, K, N

// Shared memory tiles for A and B (16×16 each)
var<workgroup> tile_a: array<f32, 256>; // 16×16
var<workgroup> tile_b: array<f32, 256>; // 16×16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;
    
    let row = global_id.y;  // Row in C
    let col = global_id.x;  // Column in C
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Accumulator for this thread's result
    var sum: f32 = 0.0;
    
    // Number of tiles needed to cover K dimension
    let num_tiles = (K + 15u) / 16u;
    
    // Iterate over tiles
    for (var tile: u32 = 0u; tile < num_tiles; tile++) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = tile * 16u + local_col;
        if (a_row < M && a_col < K) {
            tile_a[local_row * 16u + local_col] = matrix_a[a_row * K + a_col];
        } else {
            tile_a[local_row * 16u + local_col] = 0.0;
        }
        
        // Load tile of B into shared memory
        let b_row = tile * 16u + local_row;
        let b_col = col;
        if (b_row < K && b_col < N) {
            tile_b[local_row * 16u + local_col] = matrix_b[b_row * N + b_col];
        } else {
            tile_b[local_row * 16u + local_col] = 0.0;
        }
        
        // Synchronize to ensure all threads have loaded their data
        workgroupBarrier();
        
        // Compute partial dot product using shared memory
        for (var k: u32 = 0u; k < 16u; k++) {
            sum += tile_a[local_row * 16u + k] * tile_b[k * 16u + local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        matrix_c[row * N + col] = sum;
    }
}
