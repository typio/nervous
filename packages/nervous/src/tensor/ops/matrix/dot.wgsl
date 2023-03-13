// https://web.dev/gpu-compute/

struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> aMatrix: Matrix;
@group(0) @binding(1) var<storage, read> mMatrix:  Matrix;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= max(1, u32(aMatrix.s.z)) || global_id.y >= max(1, u32(mMatrix.s.w)) {
        return;
    }

    outMatrix.s = vec4(0., 0., aMatrix.s.z, mMatrix.s.w);

    let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < u32(aMatrix.s.w); i = i + 1u) {
        let a: u32 = i + outIndex.x * u32(aMatrix.s.w);
        let b: u32 = outIndex.y + i * u32(mMatrix.s.w);
        sum += aMatrix.v[a] * mMatrix.v[b];
    }

    outMatrix.v[outIndex.y + outIndex.x * u32(mMatrix.s.w)] = sum;
}
