struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= max(1, u32(a.s.z)) || global_id.y >= max(1, u32(a.s.w)) {
        return;
    }

    // TODO: try to speed up with shared memory
    o.s = vec4(0, 0, a.s[3], a.s[2]);
    o.v[global_id.x + u32(a.s[2]) * global_id.y] = a.v[global_id.x * u32(a.s[3]) + global_id.y];
}
