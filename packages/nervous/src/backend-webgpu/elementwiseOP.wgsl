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
  // does this slow program down? will it be computed once or every time? test doesn't indicate significant slowdown 🤷
  o.s = vec4(max(a.s.x, b.s.x), max(a.s.y, b.s.y), max(a.s.z, b.s.z), max(a.s.w, b.s.w));

  // this is unfortunate
  if global_id.x > u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w)) {
      return;
  }

  let lI = f32(global_id.x);

  // var i = f32(lI) % o_s[3];
  // var j = (f32(lI) / o_s[3]) % o_s[2];
  // var k = (f32(lI) / (o_s[3] * o_s[2])) % o_s[1];
  // var l = (f32(lI) / (o_s[3] * o_s[2] * o_s[1])) % o_s[0];

//   i = i % a_s[3];
  // j = j % a_s[2];
  // k = k % a_s[1];
  // l = l % a_s[0];

  // var aV: f32;
  // let aI =  l * a_s[3] * a_s[2] * a_s[1] + k * a_s[2] * a_s[1] + j * a_s[1] + i ;

  // let aI = lI % (a.s.w * (a.s.z / a_s.w));

  var aV: f32;
  var aI: u32;
  if (a.s.w <= 1 && a.s.z <= 1) {
    aI = 0u;
  } else if (a.s.w <= 1) {
    aI = u32(lI / o.s.w);
  } else if (a.s.z <= 1) {
    aI = u32(lI % o.s.w);
  } else {
    aI = u32(lI);
  }
  aV = a.v[aI];


  var bV: f32;
  var bI: u32;
  if (b.s.w <= 1 && b.s.z <= 1) {
    bI = 0u;
  } else if (b.s.w <= 1) {
    bI = u32(lI / o.s.w);
  } else if (b.s.z <= 1) {
    bI = u32(lI % o.s.w);
  } else {
    bI = u32(lI);
  }
  bV = b.v[bI];
  bV = b.v[u32(bI)];
  // A[i][j][k] = B + W *(M * N(i-x) + N *(j-y) + (k-z))

  switch flag {
      case 0u: {
          o.v[u32(lI)] = aV + bV;
      }
      case 1u: {
          o.v[u32(lI)] = aV - bV;
      }
      case 2u: {
          o.v[u32(lI)] = aV * bV;
      }
      case 3u: {
          o.v[u32(lI)] = aV / bV;
      }
      case 4u: {
          o.v[u32(lI)] = aV % bV;
      }
      default: {
          o.v[u32(lI)] = 0.;
      }
  }
}

