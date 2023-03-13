import { toNested, calcShape, flatLengthFromShape, padShape, toArr } from './tensorUtils'
import { gpuDevice } from '..'

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]

export enum BinaryOp {
    add = 0,
    minus,
    mul,
    div,
    mod,
}

export enum ScalarElementwiseOP {
    log = 0,
    pow = 1,
    applyMax = 2,
    applyMin = 3,
    exp = 4,
}

export enum ReduceOP {
    sum = 0,
    argmax,
    argmin,
}

export class Tensor {
    /**  first 4 values are shape, most to least significant dimensions, left-padded with 0's, rest are tensor values */
    readonly data: Float32Array

    readonly onGPU: boolean = false
    readonly gpuBuffer: GPUBuffer
    readonly gpuBufferShape: number[]

    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]) {
        let _shape: number[] = []
        let _values: number[] = []

        if (values.constructor === Float32Array) {
            if (shape !== undefined) console.warn('Shape was not used in tensor() call with a Float32Array')
            this.data = values
            return
        } else if (values.constructor === GPUBuffer) {
            this.onGPU = true
            this.gpuBuffer = values
            this.gpuBufferShape = padShape(shape)
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

        this.data = new Float32Array([..._shape, ..._values])
    }

    toJS = async (): Promise<Tensor> => {
        let bufferSize = Math.max(32, this.gpuBuffer.size)

        const readGPUBuffer = gpuDevice.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const commandEncoder = gpuDevice.createCommandEncoder()
        commandEncoder.copyBufferToBuffer(this.gpuBuffer, 0, readGPUBuffer, 0, bufferSize)
        gpuDevice.queue.submit([commandEncoder.finish()])
        await readGPUBuffer.mapAsync(GPUMapMode.READ)

        let result = new Float32Array(readGPUBuffer.getMappedRange())

        // buffer may have been right padded to make minimum size, undo that
        let tensorSize = 4 + flatLengthFromShape(toArr(result.slice(0, 4)))
        if (bufferSize > tensorSize * Float32Array.BYTES_PER_ELEMENT) result = result.slice(0, tensorSize)

        return new Tensor(result)
    }

    toGPU = (): Tensor => {
        const buffer = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: Math.max(32, this.data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        })
        new Float32Array(buffer.getMappedRange()).set(this.data)
        buffer.unmap()

        return new Tensor(buffer, this.shape())
    }

    // select = (dim: number, index: number) => {
    //     throw new Error('Not implemented')
    // }

    /** returns nested number array of tensor values, returns type number if scalar */
    values = async (decimals?: number) => {
        let a: Tensor = this.onGPU ? await this.toJS() : this

        if (a.data.length === 5) return a.data[4]
        return toNested(toArr(a.data, decimals, 4), a.shape())
    }

    /** returns flat tensor values */
    flatValues = async (decimals?: number) => {
        let a: Tensor = this.gpuBuffer ? await this.toJS() : this
        return toArr(a.data, decimals, 4)
    }

    print = async (decimals?: number) => {
        console.log(JSON.stringify(await this.values(decimals)).replace(/],/g, '],\n '))
    }

    /** returns tensor rank */
    rank = () => {
        let shape: number[]
        if (this.onGPU) {
            shape = this.gpuBufferShape
        } else {
            shape = toArr(this.data.slice(0, 4))
        }

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
        if (this.onGPU) {
            shape = this.gpuBufferShape
        } else {
            shape = toArr(this.data.slice(0, 4))
        }

        let i = 0
        while (i < 3 && shape[i] === 0) i++

        return shape.slice(i, 4)
    }


    /** Reshape tensor into provided shape */
    reshape = (shape: number[]) => backend.default.reshape(this, shape)

    /** Repeat tensor along dimensions */
    repeat = (scales: number[]) => backend.default.repeat(this, scales)

    /** switch rows and columns of a >=2d Tensor */
    transpose = () => backend.default.transpose(this)

    /** create tensor of dot product */
    dot = (m: Tensor) => backend.default.dot(this, m)

    inverse = () => {
        throw new Error('Not impl., maybe ever')
    }

    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul = (m: Tensor | number) => backend.default.mul(this, m)

    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div = (d: Tensor | number) => backend.default.div(this, d)

    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add = (b: Tensor | number) => backend.default.add(this, b)

    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus = (s: Tensor | number) => backend.default.minus(this, s)

    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod = (m: Tensor | number) => backend.default.mod(this, m)

    /** create tensor with relu done to all values  */
    pow = (exp: number) => backend.default.pow(this, exp)

    // broadcast(func: (any)) {
    //     let newV = []
    //     for (let i = 0; i < this.values.length; i++) {
    //         newV[i] = func(this.values[i])
    //     }
    //     return new Tensor(newV, this.shape)
    // }

    /** create tensor with sigmoid done to all values  */
    sigmoid = () => backend.default.sigmoid(this)

    /** create tensor with softplus done to all values  */
    softplus = () => backend.default.softplus(this)

    // round(decimals: number) {
    //     return this.broadcast((n: number) => Math.floor(n * (10 ** decimals)) / 10 ** decimals)
    // }

    // return softmax
    softmax = (dim) => backend.default.softmax(this, dim)

    /** create tensor with relu done to all values  */
    reLU = () => backend.default.reLU(this)

    /** create tensor with relu done to all values  */
    gradientReLU = (b: Tensor) => backend.default.gradientReLU(this, b)

    /** create tensor of exponentials of all values on e, or given base  */
    exp = (base?: number) => backend.default.exp(this, base)

    /** create tensor of log on all values */
    log = (base: number) => backend.default.log(this, base)

    /** get the mean of all values */
    mean = () => backend.default.mean(this)

    /** return the lp norm as number, default p is 2  */
    lpNorm = (p?: number) => backend.default.lpNorm(this, p)

    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm = () => backend.default.fNorm(this)

    /** returns sum of diagonal elements as number */
    trace = () => backend.default.trace(this)

    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum = (axis?: 0 | 1) => backend.default.sum(this, axis)

    compare = (b: Tensor, axis: 0 | 1) => backend.default.compare(this, b, axis)

    /** returns tensor with elementwise max of old value vs input number */
    applyMax = (n: number) => backend.default.applyMax(this, n)

    /** returns tensor with elementwise min of old value vs input number */
    applyMin = (n: number) => backend.default.applyMin(this, n)

    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax = (axis?: 0 | 1) => backend.default.getmax(this, axis)

    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin = (axis?: 0 | 1) => backend.default.getmin(this, axis)

    argmax = (axis?: 0 | 1) => backend.default.argmax(this, axis)

    argmin = (axis?: 0 | 1) => backend.default.argmin(this, axis)
}
