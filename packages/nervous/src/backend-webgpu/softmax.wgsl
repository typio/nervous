struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> dim:  u32;
@group(0) @binding(2) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(a.s[3 - dim])) {
        return;
    }

    let row_n = u32(a.s[2]);
    let col_n = u32(a.s[3]);

    o.s = a.s;

    var sum :f32 = 0.0;

    var max = -0x1p-126f;
    if (dim == 1) {
        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            let v = a.v[idx];
            if (v > max) {
                max = v;
            }
        }

        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            sum += exp(a.v[idx] - max);
        }

        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            o.v[idx] = exp(a.v[idx] - max) / sum;
        }
    } else if (dim == 0) {
        for (var i = 0u; i < row_n; i++) {
            let idx = global_id.x + col_n * i;
            let v = a.v[idx];
            if (v > max) {
                max = v;
            }
        }

        for (var i = 0u; i < row_n; i++) {
            let idx = u32(global_id.x + col_n * i);
            sum += exp(a.v[idx] - max);
        }
        
        for (var i = 0u; i < row_n; i++) {
            let idx = u32(global_id.x + col_n * i);
            o.v[idx] = exp(a.v[idx] - max) / sum;
        }
    }
}
