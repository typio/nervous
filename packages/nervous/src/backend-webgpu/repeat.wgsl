struct Matrix {
  shape: vec4<f32>,
  values: array<f32>
}

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> scales: vec4<f32>;
@group(0) @binding(2) var<storage, read_write> outMatrix: Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  outMatrix.shape = a.shape * scales;

  let li = global_id.x;

  i = li % dims[3];
  j = (li / dims[3]) % dims[2];
  k = (li / (dims[3] * dims[2])) % dims[1];
  l = (li / (dims[3] * dims[2] * dims[1])) % dims[0];

  outMatrix.values[i] = 1.;
}
