import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

export default {
  matrixStruct: wgsl`
    struct Matrix {
      shape: vec4<f32>,
      values: array<f32>
    };
    `,
}

