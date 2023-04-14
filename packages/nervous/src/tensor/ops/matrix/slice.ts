import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader } from '../../../webGPU/_index'

import { Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape, padShape } from '../../tensorUtils'

/**
 * Slice a tensor along a specified dimension.
 * @param {Tensor} a - Input tensor
 */
 export function slice(a: Tensor): Tensor {
 }
