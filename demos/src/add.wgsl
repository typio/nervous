[[block]]struct Shape {
    shape: vec2<u32>;
};

[[block]]struct Values {
    values: array<f32>;
};

[[group(0), binding(0)]] var<storage> aShape: [[access(read)]] Shape;
[[group(0), binding(1)]] var<storage> bShape: [[access(read)]] Shape;
[[group(0), binding(2)]] var<storage> aValues: [[access(read)]] Values;
[[group(0), binding(3)]] var<storage> bValues: [[access(read)]] Values;
[[group(0), binding(4)]] var<storage> outValues: [[access(write)]] Values;


[[stage(compute)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let index : u32 = global_id.x + global_id.y * aShape.shape.x;
    
}