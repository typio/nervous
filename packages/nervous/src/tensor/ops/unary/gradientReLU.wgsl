struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> b:  Matrix;
@group(0) @binding(2) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    o.s = a.s;

    if global_id.x > u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w)) {
        return;
    }

    if (b.v[global_id.x] <= 0) {
        o.v[global_id.x] = 0.;
    } else {
        o.v[global_id.x] = a.v[global_id.x];
    }
}
