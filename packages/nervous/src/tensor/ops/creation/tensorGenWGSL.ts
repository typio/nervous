import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

import snips from './snips'

export const wgslTensorGen = (fn, params) => {
  switch (fn) {
    case 'fill':
      return fillWGSL
    case 'diag':
      return diagWGSL
    case 'random':
      return randomWGSL
  }
}

// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
// "quick and dirty LCG which has a period of 2^32"
// this feels a little slow, is there an adequate faster solution?

const randomWGSL = wgsl`
  ${snips.matrixStruct}

  @group(0) @binding(0) var<storage, read> shape: vec4<f32>;
  @group(0) @binding(1) var<storage, read> seedInput: u32;
  @group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let seed = seedInput * (global_id.x + 1u) * 1099087573u;
      let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
      outMatrix.shape = shape;

      var z1: u32;
      var z2: u32;
      var z3: u32;
      var z4: u32;

      z1 = tau_step(seed, 13u, 19u, 12u, 429496729u);
      z2 = tau_step(seed, 2u, 25u, 4u, 429496729u);
      z3 = tau_step(seed, 3u, 19u, 17u, 429496729u);
      z4 = 1664525u * seed + 1013904223u;
      let r0: f32 = f32(z1 ^ z2 ^ z3 ^ z4);

      z1 = tau_step(seed, 13u, 19u, 12u, 429496729u);
      z2 = tau_step(seed, 2u, 25u, 4u, 429496729u);
      z3 = tau_step(seed, 3u, 19u, 17u, 429496729u);
      z4 = 1664525u * u32(r0) + 1013904223u;
      let r1: f32 = f32(z1 ^ z2 ^ z3 ^ z4);

      z1 = tau_step(seed, 13u, 19u, 12u, 429496729u);
      z2 = tau_step(seed, 2u, 25u, 4u, 429496729u);
      z3 = tau_step(seed, 3u, 19u, 17u, 429496729u);
      z4 = 1664525u * u32(r1) + 1013904223u;
      let r2: f32 = f32(z1 ^ z2 ^ z3 ^ z4);

      z1 = tau_step(seed, 13u, 19u, 12u, 429496729u);
      z2 = tau_step(seed, 2u, 25u, 4u, 429496729u);
      z3 = tau_step(seed, 3u, 19u, 17u, 429496729u);
      z4 = 1664525u * u32(r2) + 1013904223u;
      let r3: f32 = f32(z1 ^ z2 ^ z3 ^ z4);

      outMatrix.values[outIndex.y + outIndex.x * 4u ] = r0 * 2.3283064365387e-10;
      outMatrix.values[outIndex.y + outIndex.x * 4u + 1u ] = r1 * 2.3283064365387e-10;
      outMatrix.values[outIndex.y + outIndex.x * 4u + 2u ] = r2 * 2.3283064365387e-10;
      outMatrix.values[outIndex.y + outIndex.x * 4u + 3u ] = r3 * 2.3283064365387e-10;
  }

  fn tau_step(z: u32, s1: u32, s2: u32, s3: u32, M: u32) -> u32 {
      let b = (((z << s1) ^ z) >> s2);
      let res = (((z & M) << s3) ^ b);
      return res;
  }
`
