// https://web.dev/gpu-compute/

struct Matrix {
    shape: vec4<f32>,
    values: array<f32>
};

@group(0) @binding(0) var<storage, read> aMatrix: Matrix;
@group(0) @binding(1) var<storage, read> mMatrix:  Matrix;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    outMatrix.shape = vec4(aMatrix.shape.x, mMatrix.shape.y, 0., 0.);

    let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < u32(aMatrix.shape.y); i = i + 1u) {
        let a: u32 = i + outIndex.x * u32(aMatrix.shape.y);
        let b: u32 = outIndex.y + i * u32(mMatrix.shape.y);
        sum += aMatrix.values[a] * mMatrix.values[b];
    }

    outMatrix.values[outIndex.y + outIndex.x * u32(mMatrix.shape.y)] = sum;
}