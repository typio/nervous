import { backend } from "."
import { tensor } from "./backend-js/tensor"
import { calcShape, toArr, flatLengthFromShape, toNested } from "./tensorUtils"

// TODO: CHANGE new Array()'s to Float32Array's

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]

export type BinaryOp = "add" | "sub" | "mul" | "div" | "mod"

export class Tensor {
    readonly values: Float32Array = new Float32Array(0)
    readonly rank: 0 | 1 | 2 | 3 | 4 = 0
    readonly shape: number[] = [0]

    constructor(values: number | Rank1To4Array, shape?: number[]) {
        if (values !== undefined && shape === undefined) {
            if (values.constructor === Number) { // if scalar
                values = [values] // store scalar number as number[]
                this.shape = [1]
                this.rank = 0
            } else {
                this.shape = calcShape(values)
                this.rank = this.shape.length as typeof this.rank
            }
            if (values.constructor === Array) {
                let flatValues = values.flat() as number[]
                this.values = new Float32Array(flatValues)
            } else if (values.constructor === Float32Array) {
                this.values = values
            }

        } else if (values !== undefined && shape !== undefined) {
            if (values.constructor === Array && values[0].constructor === Array)
                throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

            // @ts-ignore: I checked the type
            let flatValues: number[] | Float32Array = values

            if (flatLengthFromShape(shape) !== flatValues.length)
                throw new Error("Values don't fit into shape.")

            this.shape = shape
            this.rank = shape.length as typeof this.rank // good ts? 🤔
            this.values = new Float32Array(flatValues)
        }
    }

    select = (dim: number, index: number) => {
        throw new Error("Not implemented")
    }

    /** return nested number array of tensor values, returns type number if scalar */
    getValues = (decimals?: number) => backend.default.getValues(this, decimals)

    /** return flat tensor values */
    getFlatValues = (decimals?: number) => backend.default.getFlatValues(this, decimals)


    /** Reshape tensor into provided shape */
    reshape = (shape: number[]) => backend.default.reshape(this, shape)


    /** switch rows and columns of a >=2d Tensor */
    transpose = () => backend.default.transpose(this)

    /** create tensor of dot product */
    matmul = (m: Tensor | number) => backend.default.matmul(this, m)

    inverse = () => {
        throw new Error("Not impl., maybe ever")
    }

    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul = (m: Tensor | number, axis?: number) => backend.default.mul(this, m, axis);

    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div = (d: Tensor | number, axis?: number) => backend.default.div(this, d, axis);

    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add = (b: number | Tensor, axis?: number): Tensor => backend.default.add(this, b, axis);

    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus = (s: number | Tensor, axis?: number) => backend.default.minus(this, s, axis);

    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod = (m: number | Tensor, axis?: number) => backend.default.mod(this, m, axis);


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