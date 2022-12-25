// struct Shape {
//     shape: vec2<u32>
// };

struct Values {
    values: array<f32>
};

// @group(0) @binding(0) var<storage, read> aShape:  Shape;
// @group(0) @binding(1) var<storage, read> bShape:  Shape;
@group(0) @binding(0) var<storage, read> aValues: Values;
@group(0) @binding(1) var<storage, read> bValues:  Values;
@group(0) @binding(2) var<storage, read_write> outValues:  Values;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index: u32 = global_id.x;
    outValues.values[index] = aValues.values[index] + bValues.values[index];
}