import { backend } from '.'
import { calcShape, flatLengthFromShape } from './tensorUtils'

// TODO: CHANGE new Array()'s to Float32Array's

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]

export enum BinaryOp {
  add = 0,
  minus,
  mul,
  div,
  mod,
}

export class Tensor {
  /**  first 4 values are shape (right padded 0s), rest are tensor values */
  readonly data: Float32Array

  readonly usingGPUBuffer: boolean = false
  readonly webGPUBuffer: any
  readonly webGPUBufferShape: number[]

  /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
   * or pass raw Float32Array already in internal Tensor data form. */
  constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]) {
    let _shape: number[] = []
    let _values: number[] = []

    if (values.constructor === Float32Array) {
      this.data = values
      return
    } else if (values.constructor !== Number && values.constructor !== Array) {
      // is GPUBuffer, can't use "GPUBuffer" bc @webgpu/types DOESN'T WORK!
      this.usingGPUBuffer = true
      this.webGPUBuffer = values
      this.webGPUBufferShape = shape
      return
    }

    if (values !== undefined && shape === undefined) {
      if (values.constructor === Number) {
        // if scalar
        values = [values] // store scalar number as number[]
        _shape = [1, 1] // questionable
      } else {
        // @ts-ignore: I checked the type
        _shape = calcShape(values)
      }
      if (values.constructor === Array) {
        let flatValues = values.flat() as number[]
        _values = flatValues
      }
    } else if (values !== undefined && shape !== undefined) {
      if (values.constructor === Array && values[0].constructor === Array)
        throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

      let flatValues: number[] = Array.prototype.slice.call(values)

      if (flatLengthFromShape(shape) !== flatValues.length) throw new Error("Values don't fit into shape.")

      _shape = shape
      _values = flatValues
    }

    while (_shape.length < 4) {
      _shape.push(0)
    }

    this.data = new Float32Array([..._shape, ..._values])
  }

  toJS = async (): Promise<Tensor> => backend.default.toJS(this)

  toGPU = async (): Promise<Tensor> => backend.default.toGPU(this)

  select = async (dim: number, index: number) => {
    throw new Error('Not implemented')
  }

  /** returns nested number array of tensor values, returns type number if scalar */
  values = async (decimals?: number): Promise<number[] | number> => backend.default.values(this, decimals)

  /** returns flat tensor values */
  flatValues = async (decimals?: number): Promise<number[]> => backend.default.flatValues(this, decimals)

  /** returns tensor rank */
  rank = async () => backend.default.rank(this)

  /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
  shape = async () => backend.default.shape(this)

  /** Reshape tensor into provided shape */
  reshape = async (shape: number[]) => backend.default.reshape(this, shape)

  /** switch rows and columns of a >=2d Tensor */
  transpose = async () => backend.default.transpose(this)

  /** create tensor of dot product */
  matmul = async (m: Tensor) => backend.default.matmul(this, m)

  inverse = async () => {
    throw new Error('Not impl., maybe ever')
  }

  /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
  mul = async (m: Tensor | number) => backend.default.mul(this, m)

  /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
  div = async (d: Tensor | number) => backend.default.div(this, d)

  /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
  add = async (b: Tensor | number): Promise<Tensor> => backend.default.add(this, b)

  /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
  minus = async (s: Tensor | number) => backend.default.minus(this, s)

  /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
  mod = async (m: Tensor | number) => backend.default.mod(this, m)

  /** create tensor with relu done to all values  */
  pow = async (exp: number) => backend.default.pow(this, exp)

  // broadcast(func: (any)) {
  //     let newV = []
  //     for (let i = 0; i < this.values.length; i++) {
  //         newV[i] = func(this.values[i])
  //     }
  //     return new Tensor(newV, this.shape)
  // }

  /** create tensor with sigmoid done to all values  */
  sigmoid = async () => backend.default.sigmoid(this)

  /** create tensor with softplus done to all values  */
  softplus = async () => backend.default.softplus(this)

  // round(decimals: number) {
  //     return this.broadcast((n: number) => Math.floor(n * (10 ** decimals)) / 10 ** decimals)
  // }

  // return softmax
  softmax = async () => backend.default.softmax(this)

  /** create tensor with relu done to all values  */
  reLU = async () => backend.default.reLU(this)

  /** create tensor with relu done to all values  */
  gradientReLU = async () => backend.default.gradientReLU(this)

  /** create tensor of exponentials of all values on e, or given base  */
  exp = async (base?: number) => backend.default.exp(this, base)

  /** create tensor of log on all values */
  log = async () => backend.default.log(this)

  /** get the mean of all values */
  mean = async () => backend.default.mean(this)

  /** return the lp norm as number, default p is 2  */
  lpNorm = async (p?: number): Promise<number> => backend.default.lpNorm(this, p)

  /** return Frobenius Norm as number, represents the size of a matrix */
  fNorm = async () => backend.default.fNorm(this)

  /** returns sum of diagonal elements as number */
  trace = async () => backend.default.trace(this)

  /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
  sum = async (axis?: 0 | 1): Promise<Tensor> => backend.default.sum(this, axis)

  /** returns tensor with elementwise max of old value vs input number */
  applymax = async (n: number) => backend.default.applymax(this, n)

  /** returns tensor with elementwise min of old value vs input number */
  applymin = async (n: number) => backend.default.applymin(this, n)

  /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
  getmax = async (axis?: 0 | 1) => backend.default.getmax(this, axis)

  /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
  getmin = async (axis?: 0 | 1) => backend.default.getmin(this, axis)

  argmax = async () => backend.default.argmax(this)

  argmin = async () => backend.default.argmin(this)
}
