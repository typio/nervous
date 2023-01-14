struct Matrix {
    shape: vec4<f32>,
    values: array<f32>
};

@group(0) @binding(0) var<storage, read> shape: vec4<f32>;
@group(0) @binding(1) var<storage, read> value: f32;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let thing = 2;
  outMatrix.shape = shape;
  outMatrix.values[global_id.x] = value;
}
