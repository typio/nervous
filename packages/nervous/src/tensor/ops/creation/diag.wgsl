struct Matrix {
    shape: vec4<f32>,
    values: array<f32>
};

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> shape: vec4<f32>;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let side_length = u32(shape.z);
  if (global_id.x > side_length) {
    return;
  }

  outMatrix.shape = shape;

  // relies on the array buffer initially being filled with 0's
  outMatrix.values[global_id.x * (side_length + 1)] = values[global_id.x];
}
