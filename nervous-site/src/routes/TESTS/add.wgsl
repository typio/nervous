
// struct Shape {
//     shape: vec2<u32>
// };

struct Matrix {
    shape: vec4<f32>,
    values: array<f32>
};

// @group(0) @binding(0) var<storage, read> aShape:  Shape;
// @group(0) @binding(1) var<storage, read> bShape:  Shape;
@group(0) @binding(0) var<storage, read> aMatrix: Matrix;
@group(0) @binding(1) var<storage, read> bMatrix:  Matrix;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    outMatrix.shape = aMatrix.shape;

    let index: u32 = global_id.x;
    outMatrix.values[index] = aMatrix.values[index] + bMatrix.values[index];
}