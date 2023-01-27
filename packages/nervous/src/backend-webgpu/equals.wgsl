struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> m:  Matrix;
@group(0) @binding(2) var<storage, read_write> o: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= max(1, u32(a.s.z)) * max(1, u32(a.s.w)) {
        return;
    }
    o = 1u;

    // this seems to work though I wouldn't expect it to
    // is it a sign that I'm excessively looping this stuff?
    if (a.v[global_id.x] != m.v[global_id.x]) {
        o = 0u;
    }

}
