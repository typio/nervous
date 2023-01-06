import { backend } from "."
import { calcShape, flatLengthFromShape } from "./tensorUtils"

// TODO: CHANGE new Array()'s to Float32Array's

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]

export enum BinaryOp {
    add = 0,
    minus,
    mul,
    div,
    mod
}

export class Tensor {
    /**  first 4 values are shape (right padded 0s), rest are tensor values */
    readonly data: Float32Array

    readonly usingGPUBuffer: boolean = false;
    readonly webGPUBuffer: any;
    readonly webGPUBufferShape: number[]

    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested, 
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]) {
        let _shape: number[] = []
        let _values: number[] = []

        if (values.constructor === Float32Array) {
            this.data = values
            return
        } else if (values.constructor !== Number && values.constructor !== Array) { // is GPUBuffer, can't use "GPUBuffer" bc @webgpu/types DOESN'T WORK!
            this.usingGPUBuffer = true
            this.webGPUBuffer = values
            this.webGPUBufferShape = shape
            return
        }

        if (values !== undefined && shape === undefined) {
            if (values.constructor === Number) { // if scalar
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

            if (flatLengthFromShape(shape) !== flatValues.length)
                throw new Error("Values don't fit into shape.")

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

    select = (dim: number, index: number) => {
        throw new Error("Not implemented")
    }

    /** returns nested number array of tensor values, returns type number if scalar */
    values = (decimals?: number): number[] | number => backend.default.values(this, decimals)

    /** returns flat tensor values */
    flatValues = (decimals?: number): number[] => backend.default.flatValues(this, decimals)

    /** returns tensor rank */
    rank = () => backend.default.rank(this)

    /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
    shape = () => backend.default.shape(this)


    /** Reshape tensor into provided shape */
    reshape = (shape: number[]) => backend.default.reshape(this, shape)


    /** switch rows and columns of a >=2d Tensor */
    transpose = () => backend.default.transpose(this)

    /** create tensor of dot product */
    matmul = (m: Tensor) => backend.default.matmul(this, m)

    inverse = () => {
        throw new Error("Not impl., maybe ever")
    }

    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul = (m: Tensor | number) => backend.default.mul(this, m);

    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div = (d: Tensor | number) => backend.default.div(this, d);

    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add = (b: Tensor | number): Tensor => backend.default.add(this, b);

    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus = (s: Tensor | number) => backend.default.minus(this, s);

    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod = (m: Tensor | number) => backend.default.mod(this, m);


    /** create tensor with relu done to all values  */
    pow = (exp: number) => backend.default.pow(this, exp);


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
    softmax = () => backend.default.softmax(this)

    /** create tensor with relu done to all values  */
    reLU = () => backend.default.reLU(this)

    /** create tensor with relu done to all values  */
    gradientReLU = () => backend.default.gradientReLU(this)

    /** create tensor of exponentials of all values on e, or given base  */
    exp = (base?: number) => backend.default.exp(this, base)

    /** create tensor of log on all values */
    log = () => backend.default.log(this)

    /** get the mean of all values */
    mean = () => backend.default.mean(this)

    /** return the lp norm as number, default p is 2  */
    lpNorm = (p?: number): number => backend.default.lpNorm(this, p)

    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm = () => backend.default.fNorm(this)

    /** returns sum of diagonal elements as number */
    trace = () => backend.default.trace(this)

    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum = (axis?: 0 | 1): Tensor => backend.default.sum(this, axis)

    /** returns tensor with elementwise max of old value vs input number */
    applymax = (n: number) => backend.default.applymax(this, n)

    /** returns tensor with elementwise min of old value vs input number */
    applymin = (n: number) => backend.default.applymin(this, n)

    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax = (axis?: 0 | 1) => backend.default.getmax(this, axis)

    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin = (axis?: 0 | 1) => backend.default.getmin(this, axis)

    argmax = () => backend.default.argmax(this)

    argmin = () => backend.default.argmin(this)
}