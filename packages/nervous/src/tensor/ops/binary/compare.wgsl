struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> m:  Matrix;
@group(0) @binding(2) var<storage, read> axis: i32;
@group(0) @binding(3) var<storage, read_write> o: Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (axis == -1) {
        if global_id.x >= max(1, u32(a.s.z)) * max(1, u32(a.s.w)) {
            return;
        }

        o.s = a.s;
        var newV: f32 = 1.;
        if (a.v[global_id.x] != m.v[global_id.x]) {
            newV = 0;
        }
        o.v[global_id.x] = newV;
    } else if (axis == 0) { // columns are equal
        if global_id.x >= u32(a.s[3]) {
            return;
        }

        o.s = vec4(0., 0., 1., a.s[3]);
        var newV: f32 = 1.;
        for (var i: u32 = 0; i < u32(a.s[2]); i += 1) {
            let idx: u32 = global_id.x + u32(a.s[3]) * i;
            if (a.v[idx] != m.v[idx]) {
                newV = 0;
            }
        }
        o.v[global_id.x] = newV;
    } else if (axis == 1) { // rows are equal
        if global_id.x >= u32(a.s[2]) {
            return;
        }

        o.s = vec4(0., 0., a.s[2], 1.);
        var newV: f32 = 1.;
        for (var i: u32 = 0; i < u32(a.s[3]); i += 1) {
            let idx: u32 = global_id.x * u32(a.s[3]) + i;
            if (a.v[idx] != m.v[idx]) {
                newV = 0;
            }
        }
        o.v[global_id.x] = newV;
    }
}
