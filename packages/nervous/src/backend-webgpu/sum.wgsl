struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> axis:  i32;
@group(0) @binding(2) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (axis == -1 && global_id.x >= 1) {
        return;
    } else if (global_id.x >= u32(a.s[3 - axis])) {
        return;
    }

    let row_n = u32(a.s[2]);
    let col_n = u32(a.s[3]);

    if (axis == 1) {
        o.s = vec4(0,0,a.s[2],1);
    } else if (axis == 0) {
        o.s = vec4(0,0,1,a.s[3]);
    } else {
        o.s = vec4(0, 0, 0, 1);
    }

    var sum :f32 = 0.0;

    if (axis == 1) {
        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            sum += a.v[idx];
        }
        o.v[global_id.x] = sum;
    } else if (axis == 0) {
        for (var i = 0u; i < row_n; i++) {
            let idx = u32(global_id.x + col_n * i);
            sum += a.v[idx];
        }
        o.v[global_id.x] = sum;
    } else {
        // unfortunately is signle threaded
        let length = u32(max(1., a.s.x) * max(1., a.s.y) * max(1., a.s.z) * max(1., a.s.w));
        for (var i = 0u; i < length; i++){
            sum += a.v[i];
        }
        o.v[0] = sum;
    }
}
