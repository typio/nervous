struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> b:  Matrix;
@group(0) @binding(2) var<storage, read> flag: u32;
@group(0) @binding(3) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // does this slow program down? will it be computed once or every time? test doesn't indicate significant slowdown 🤷‍♂️
    o.s.x = max(a.s.x, b.s.x);
    o.s.y = max(a.s.y, b.s.y);
    o.s.z = max(a.s.z, b.s.z);
    o.s.w = max(a.s.w, b.s.w);

    // this is nuts
    if global_id.x > u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w)) {
        return;
    }

    let i: u32 = global_id.x;

    var bV: f32;
    if b.s.x == 1. || b.s.y == 1. {
        // this formula only works for a 1d column or row vector, it's gotta be slow
        let bX = (i / u32(o.s.y));
        let bY = (i % u32(o.s.y));
        let bI = ((bX * u32(o.s.y)) % u32(b.s.x)) + bY % u32(b.s.y);
        bV = b.v[bI];
    } else {
        bV = b.v[i];
    }

    var aV: f32;
    if a.s.x == 1. || a.s.y == 1. {
        let aX = (i / u32(o.s.y));
        let aY = (i % u32(o.s.y));
        let aI = ((aX * u32(o.s.y)) % u32(a.s.x)) + aY % u32(a.s.y);
        aV = a.v[aI];
    } else {
        aV = a.v[i];
    }

    switch flag {
        case 0u: {
            o.v[i] = aV + bV;
        }
        case 1u: {
            o.v[i] = aV - bV;
        }
        case 2u: {
            o.v[i] = aV * bV;
        }
        case 3u: {
            o.v[i] = aV / bV;
        }
        case 4u: {
            o.v[i] = aV % bV;
        }
        default: {
            o.v[i] = 0.;
        }
    }
}

