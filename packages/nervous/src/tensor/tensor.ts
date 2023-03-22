import { toNested, calcShape, flatLengthFromShape, padShape, toArr, unpadShape } from './tensorUtils'
import { gpuDevice } from '..'

import { unaryOp } from './ops/unary/_index'
import { binaryOp } from './ops/binary/_index'
import { reductionOp } from './ops/reduction/_index'
import { dot, transpose } from './ops/matrix/_index'

export type Rank1To4Array = number[] | number[][] | number[][][] | number[][][][]

export enum UnaryOp {
    log = 0,
    exp,

    relu,
    leakyRelu,

    softmax
}

export enum BinaryOp {
    add = 0,
    minus,
    mul,
    div,
    mod,
    pow,
    eq,
    gt,
    lt,

    gradientRelu,
    gradientLeakyRelu
}

export enum ScalarElementwiseOp {
    log = 0,
    pow,
    applyMax,
    applyMin,
    exp,
}

export enum ReductionOp {
    sum = 0,
    argmax,
    argmin,
    max,
    min,
    mean,
}

const createBuffer = (data: Float32Array): GPUBuffer => {
    const buffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, data.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    new Float32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    return buffer
}

const ensureTensor = (input: Tensor | number | Rank1To4Array): Tensor => {
    if (input instanceof Tensor) {
        return input
    } else {
        return new Tensor(input)
    }
}

export class Tensor {
    /**  first 4 values are shape, most to least significant dimensions, left-padded with 0's, rest are tensor values */
    readonly buffer: GPUBuffer
    /** doesn't include shape, gradient shape is defined by first 4 of buffer */
    readonly gradientBuffer: GPUBuffer | undefined
    /** util to decide if tensor OPs are legal inside JS while values are away on GPU */
    readonly tensorShape: number[]

    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]) {
        let _shape: number[] = []
        let _values: number[] = []

        if (values.constructor === Float32Array) {
            this.tensorShape = toArr(values.slice(0, 4))
            this.buffer = createBuffer(values)
            return
        } else if (values.constructor === GPUBuffer) {
            // TODO: feels like inconsistent handling of shape, expectation of input format
            this.tensorShape = shape
            this.buffer = values
            return
        }

        if (values !== undefined && shape === undefined) {
            if (values.constructor === Number) {
                // if scalar
                values = [values] // store scalar number as number[]
                _shape = [1] // questionable
            } else {
                // @ts-ignore: I checked the type
                _shape = calcShape(values)
            }
            if (values.constructor === Array) {
                // This used to only require one flat()
                _values = values.flat().flat().flat() as number[]
            }
        } else if (values !== undefined && shape !== undefined) {
            if (values.constructor === Array && values[0].constructor === Array)
                throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

            let flatValues: number[] = Array.prototype.slice.call(values)

            if (flatLengthFromShape(shape) !== flatValues.length)
                throw new Error("Values don't fit into shape.")

            _shape = shape
            _values = flatValues
        }

        _shape = padShape(_shape)

        this.buffer = createBuffer(new Float32Array([..._shape, ..._values]))
        this.tensorShape = _shape
    }

    /** returns [values, shape, gradientValues]*/
    toJS = async (): Promise<[Float32Array, Float32Array, Float32Array]> => {
        let bufferSize = Math.max(32, this.buffer.size)

        const readGPUBuffer = gpuDevice.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const commandEncoder = gpuDevice.createCommandEncoder()
        commandEncoder.copyBufferToBuffer(this.buffer, 0, readGPUBuffer, 0, bufferSize)
        gpuDevice.queue.submit([commandEncoder.finish()])
        await readGPUBuffer.mapAsync(GPUMapMode.READ)

        let result = new Float32Array(readGPUBuffer.getMappedRange())

        // buffer may have been right padded to make minimum size, undo that
        let tensorSize = 4 + flatLengthFromShape(toArr(result.slice(0, 4)))
        if (bufferSize > tensorSize * Float32Array.BYTES_PER_ELEMENT) result = result.slice(0, tensorSize)

        // TODO: gradient

        return [result.slice(4), result.slice(0, 4), new Float32Array(0)]
    }


    // select = (dim: number, index: number) => {
    //     throw new Error('Not implemented')
    // }

    /** returns nested number array of tensor values, returns type number if scalar */
    values = async (decimals?: number) => {
        let [v, s, _gV] = await this.toJS()
        if (v.length === 1) return v[0]
        return toNested(toArr(v, decimals), unpadShape(toArr(s)))
    }

    /** returns flat tensor values */
    flatValues = async (decimals?: number) => {
        let [v, _s, _gV] = await this.toJS()
        if (v.length === 1) return v[0]
        return toArr(v, decimals)
    }

    print = async (decimals?: number) => {
        console.log(JSON.stringify(await this.values(decimals)).replace(/],/g, '],\n '))
    }

    /** returns tensor rank */
    rank = () => {
        let shape: number[]
        shape = this.tensorShape

        if (shape[3] === 1 && shape[2] === 0) return 0 // scalar

        let i = 3
        while (i > 0) {
            if (shape[i - 1] === 0) {
                break
            }
            i--
        }
        return 4 - i
    }

    /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
    shape = () => {
        // remove leading 0's in shape segement of data
        let shape: number[]
        shape = this.tensorShape

        let i = 0
        while (i < 3 && shape[i] === 0) i++

        return shape.slice(i, 4)
    }

    //
    // ─── TENSOR OPERATIONS ───────────────────────────────────────────────────────────────────────
    //

    // ─── matrix ops ──────────────────────────────────────────────────────────────────────────────

    /** switch rows and columns of a >=2d Tensor */
    transpose = () => transpose(this)

    /** create tensor of dot product */
    dot = (m: Tensor) => dot(this, m)

    /** Reshape tensor into provided shape */
    // reshape = (shape: number[]) => backend.default.reshape(this, shape)

    /** Repeat tensor along dimensions */
    // repeat = (scales: number[]) => backend.default.repeat(this, scales)

    // ─── binary ops ──────────────────────────────────────────────────────────────────────────────

    /** create tensor with number a OR each value of a tensor
    * a added to each value of input tensor  */
    add = (b: Tensor | number) => binaryOp(BinaryOp.add, this, ensureTensor(b))

    /** create tensor with number m OR each value of a tensor
    * m subtracted from each value of input tensor  */
    minus = (s: Tensor | number) => binaryOp(BinaryOp.minus, this, ensureTensor(s))

    /** create tensor of elementwise matrix multiplication,
    * if using a "scalar" tensor put scalar in mul argument */
    mul = (m: Tensor | number) => binaryOp(BinaryOp.mul, this, ensureTensor(m))

    /** create tensor of elementwise matrix division,
    * if using a "scalar" tensor put scalar in div argument */
    div = (d: Tensor | number) => binaryOp(BinaryOp.div, this, ensureTensor(d))

    /** create tensor with number m OR each value of a tensor
    * m mod with each value of input tensor  */
    mod = (m: Tensor | number) => binaryOp(BinaryOp.mod, this, ensureTensor(m))

    pow = (exp: number) => binaryOp(BinaryOp.pow, this, ensureTensor(exp))

    /** to compare rows, columns, etc. use compare() a reduction operation */
    eq = (b: Tensor) => binaryOp(BinaryOp.eq, this, ensureTensor(b))

    lt = (b: Tensor) => binaryOp(BinaryOp.lt, this, ensureTensor(b))

    gt = (b: Tensor) => binaryOp(BinaryOp.gt, this, ensureTensor(b))

    gradientRelu = (b: Tensor) => binaryOp(BinaryOp.gradientRelu, this, b)

    gradientLeakyRelu = (b: Tensor) => binaryOp(BinaryOp.gradientLeakyRelu, this, b)

    // ─── unary ops ───────────────────────────────────────────────────────────────────────────────

    /** create tensor of exponentials of all values on e, or given base  */
    exp = (base?: number) => unaryOp(UnaryOp.exp, this, base)

    /** create tensor of log on all values */
    log = (base: number) => unaryOp(UnaryOp.log, this, base)

    /** returns tensor with elementwise max of old value vs input number */
    // applyMax = (n: number) => backend.default.applyMax(this, n)

    /** returns tensor with elementwise min of old value vs input number */
    // applyMin = (n: number) => backend.default.applyMin(this, n)

    /** create tensor with relu done to all values  */
    relu = () => unaryOp(UnaryOp.relu, this)

    /** create tensor with relu done to all values  */
    leakyRelu = () => unaryOp(UnaryOp.leakyRelu, this)

    /** create tensor with sigmoid done to all values  */
    // sigmoid = () => backend.default.sigmoid(this)

    /** create tensor with softplus done to all values  */
    // softplus = () => backend.default.softplus(this)

    // return softmax
    softmax = (dim: number) => unaryOp(UnaryOp.softmax, this, dim)


    // round(decimals: number) {
    //     return this.broadcast((n: number) => Math.floor(n * (10 ** decimals)) / 10 ** decimals)
    // }

    // ─── reduction ops ───────────────────────────────────────────────────────────────────────────

    argmax = (axis?: 0 | 1) => reductionOp(ReductionOp.argmax, this, axis)

    argmin = (axis?: 0 | 1) => reductionOp(ReductionOp.argmin, this, axis)

    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    max = (axis?: 0 | 1) => reductionOp(ReductionOp.max, this, axis)

    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    min = (axis?: 0 | 1) => reductionOp(ReductionOp.min, this, axis)

    /** get the mean of all values */
    mean = (axis?: 0 | 1) => reductionOp(ReductionOp.mean, this, axis)

    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum = (axis?: 0 | 1) => reductionOp(ReductionOp.sum, this, axis)
}