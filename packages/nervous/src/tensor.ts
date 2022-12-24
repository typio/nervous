import { calcShape, toArr, flatLengthFromShape, toNested } from "./tensorUtils"
import { randomNormalNumber } from "./utils"

// TODO: CHANGE new Array()'s to Float32Array's

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]

export type BinaryOp = "Add" | "Sub" | "Mul" | "Div" | "Mod"

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

    select(dim: number, index: number) {

    }

    /** return nested number array of tensor values, returns type number if scalar */
    getValues(decimals?: number) {
        if (this.rank === 0) return this.values[0]
        return toNested(toArr(this.values, decimals), this.shape)
    }

    /** return flat tensor values */
    getFlatValues(decimals?: number) {
        return toArr(this.values, decimals)
    }

    /** Reshape tensor into provided shape */
    reshape(shape: number[]) {
        return new Tensor(toArr(this.values), shape)
    }

    /** switch rows and columns of a >=2d Tensor */
    transpose() {
        if (this.rank === 0)
            return this
        if (this.rank === 1) {
            let arr = []
            for (let i = 0; i < this.values.length; i++) {
                arr[i] = this.values[i]
            }
            return new Tensor(arr, [this.shape[0], 1])
        }
        if (this.rank === 2) {
            // idiomatic 👍
            const A = this.getValues()

            let newV = new Array(this.shape[1])
            for (let i = 0; i < newV.length; i++) {
                newV[i] = new Array(this.shape[0])
                for (let j = 0; j < newV[i].length; j++)
                    newV[i][j] = 0
            }

            for (let i = 0; i < A.length; i++) {
                for (let j = 0; j < A[0].length; j++) {
                    newV[j][i] = A[i][j]
                }
            }
            return tensor(newV)
        }
        throw new Error("Transpose on tensor of rank > 2 is not yet supported.")
    }

    /** create tensor of dot product */
    matmul(m: Tensor | number) {
        if (typeof m === 'number' || this.rank === 0) {
            throw new Error("Please use Tensor.mul() for tensor scalar multiplication.")
        }

        // if 1d * 1d 
        if ((this.rank === 1 && m.rank === 1) || (this.rank === 2 && m.rank === 2 && this.shape[0] === 1 && m.shape[1] === 1)) {
            let newV: number = 0
            for (let i = 0; i < this.values.length; i++)
                newV += this.values[i] * m.values[i]
            return new Tensor(newV)
        }

        // 1d * 2d
        if ((this.rank === 1 && m.rank > 1) || (m.rank === 1 && this.rank > 1)) {
            if (this.rank === 1) {
                let newV = (new Array(m.shape[1])).fill(0)
                for (let i = 0; i < m.shape[1]; i++) {
                    for (let j = 0; j < this.shape[0]; j++) {
                        newV[i] += this.values[j] * m.values[j * m.shape[1] + i];
                    }
                }

                return new Tensor(newV)
            } else {

            }
        }

        // 2d * 2d
        if (this.rank === 2 && m.rank === 2) {
            if (this.shape[1] !== m.shape[0])
                throw new Error("Tensors not compatible shapes for multiplication.")

            let A = toNested(toArr(this.values), this.shape)
            let B = toNested(toArr(m.values), m.shape)

            let newV = new Array(this.shape[0])
            for (let i = 0; i < newV.length; i++) {
                newV[i] = new Array(m.shape[1])
                for (let j = 0; j < newV[i].length; j++)
                    newV[i][j] = 0
            }

            for (let i = 0; i < A.length; i++) {
                for (let j = 0; j < B[0].length; j++) {
                    for (let k = 0; k < B.length; k++) {
                        newV[i][j] += A[i][k] * B[k][j]
                    }
                }
            }

            return new Tensor(newV)
        }

        throw new Error("Tensor matmul on rank > 2 tensors not yet supported.")
    }

    inverse() {
        throw new Error("Not impl., maybe ever")
    }

    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul(m: Tensor | number, axis?: number) {
        return elementwiseOp(this, m, 'mul', axis)
    }

    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div(d: Tensor | number, axis?: number) {
        return elementwiseOp(this, d, 'div', axis)
    }

    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add(a: number | Tensor, axis?: number) {
        return elementwiseOp(this, a, 'add', axis)
    }

    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus(s: number | Tensor, axis?: number) {
        return elementwiseOp(this, s, 'sub', axis)
    }

    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod(m: number | Tensor, axis?: number) {
        return elementwiseOp(this, m, 'mod', axis)
    }

    /** create tensor with relu done to all values  */
    pow(exp: number) {
        return this.broadcast((x: number) => x ** exp)
    }

    broadcast(func: (any)) {
        let newV = []
        for (let i = 0; i < this.values.length; i++) {
            newV[i] = func(this.values[i])
        }
        return new Tensor(newV, this.shape)
    }

    /** create tensor with sigmoid done to all values  */
    sigmoid() {
        return this.broadcast((n: number) => 1 / (1 + Math.E ** -n))
    }

    /** create tensor with softplus done to all values  */
    softplus() {
        return this.broadcast((n: number) => Math.log(1 + Math.E ** n))
    }

    // round(decimals: number) {
    //     return this.broadcast((n: number) => Math.floor(n * (10 ** decimals)) / 10 ** decimals)
    // }

    // return softmax
    softmax() {
        let minusMaxTensor: Tensor = this
        if (this.rank === 0 || this.rank === 1) {
            minusMaxTensor = tensor(this.minus(this.getmax()).getFlatValues(), [1, this.values.length])
        } else if (this.rank === 2) {
            let newV = new Array(this.shape[0])
            for (let i = 0; i < this.shape[0]; i++) {
                let row = tensor(this.getValues()[i])
                newV[i] = row.minus(row.getmax()).getFlatValues()
            }
            minusMaxTensor = tensor(newV)
        } else throw new Error(`Softmax only supports [0-2]d tensors, yours is ${this.rank}d`)


        let outputs = (new Array(minusMaxTensor.shape[0] * minusMaxTensor.shape[1]))

        for (let j = 0; j < minusMaxTensor.shape[0]; j++) {
            let eValues = [];
            for (let i = 0; i < minusMaxTensor.shape[1]; i++) {
                eValues.push(Math.E ** minusMaxTensor.values[j * minusMaxTensor.shape[1] + i]);
            }

            let eValuesSum = 0;
            for (let i = 0; i < eValues.length; i++) {
                eValuesSum += eValues[i];
            }

            for (let i = 0; i < eValues.length; i++) {
                outputs[j * minusMaxTensor.shape[1] + i] = eValues[i] / eValuesSum
            }
        }

        return new Tensor(outputs, minusMaxTensor.shape);
    }

    /** create tensor with relu done to all values  */
    reLU() {
        return this.broadcast((x: number) => x > 0 ? x : 0)
    }

    /** create tensor with relu done to all values  */
    gradientReLU() {
        return this.broadcast((x: number) => x > 0 ? 1 : 0)
    }


    /** create tensor of exponentials of all values on e, or given base  */
    exp(base?: number) {
        if (base === undefined)
            base = Math.E

        return this.broadcast((x: number) => base ** x)
    }

    /** create tensor of log on all values */
    log() {
        return this.broadcast((x: number) => Math.log(x))
    }

    /** get the mean of all values */
    mean() {
        return (this.sum().getValues() / this.values.length)
    }

    /** return the lp norm as number, default p is 2  */
    lpNorm(p?: number): number {
        if (p !== undefined) {
            let vals = this.values
            for (let i = 0; i < vals.length; i++) {
                vals[i] = vals[i] ** p
            }
            let sum = 0
            for (let i = 0; i < vals.length; i++) {
                sum += vals[i]
            }
            return sum ** (1 / p)
        } else if (p === Infinity) { // lp norm where p === Inf simplifies to value of largest magnitude el
            let max = -Infinity
            for (let i = 0; i < this.values.length; i++)
                max = this.values[i] > max ? this.values[i] : max
            return max
        } else {
            let vals = this.values
            for (let i = 0; i < vals.length; i++) {
                vals[i] = vals[i] ** 2
            }
            let sum = 0
            for (let i = 0; i < vals.length; i++) {
                sum += vals[i]
            }
            return sum ** (1 / 2)
        }
    }

    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm() {
        let fNorm = 0
        for (let i = 0; i < this.values.length; i++)
            fNorm += this.values[i] ** 2
        return Math.sqrt(fNorm)
    }

    /** returns sum of diagonal elements as number */
    trace() {
        let shape0 = this.shape[0]
        if (this.rank === 2 && shape0 === this.shape[1]) {
            let sum = 0
            for (let i = 0; i < shape0; i++) {
                sum += this.values[i * shape0 + i]
            }
            return sum
        } else if (this.rank === 0) {
            return this.values[0]
        }
        throw new Error("Must be square 2d matrix")
    }

    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum(axis?: 0 | 1): Tensor {
        if (this.rank === 0) return this
        if (axis === 0) {
            if (this.rank === 1) return this
            if (this.rank > 2) throw new Error('Rank too high for column sum, rank is >2')

            let newV = new Array(this.shape[1]).fill(0)
            for (let i = 0; i < this.shape[1]; i++) {
                for (let j = 0; j < this.shape[0]; j++) {
                    newV[i] += this.values[i + this.shape[1] * j]

                }
            }
            return tensor(newV, [this.shape[1]])

        } else if (axis === 1) {
            if (this.rank > 2) throw new Error('Rank too high for row sum, rank is >2')

            let newV = new Array(this.shape[0]).fill(0)
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    newV[i] += this.values[i * this.shape[1] + j]

                }
            }
            return tensor(newV, [this.shape[0], 1])

        } else {
            let sum = 0
            for (let i = 0; i < this.values.length; i++)
                sum += this.values[i]
            return tensor(sum)
        }
    }

    /** returns tensor with elementwise max of old value vs input number */
    applyMax(n: number) {
        return this.broadcast((x: number) => (x > n) ? x : n)
    }

    /** returns tensor with elementwise min of old value vs input number */
    applyMin(n: number) {
        return this.broadcast((x: number) => (x < n) ? x : n)

    }

    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax(axis?: 0 | 1) {
        if (axis === undefined) {
            let currVal = this.values[0]
            let max = currVal
            for (let i = 0; i < this.values.length; i++) {
                currVal = this.values[i]
                if (currVal > max) max = currVal
            }
            return max
        }
        let newV = new Array(this.shape[1]).fill(-Infinity)
        if (axis === 0) {
            if (this.rank > 2) throw new Error('Rank too high for column max, rank is >2')
            for (let i = 0; i < this.shape[1]; i++) {
                for (let j = 0; j < this.shape[0]; j++) {
                    let currVal = this.values[i + this.shape[1] * j]
                    let oldMax = newV[i]
                    if (currVal > oldMax) newV[i] = currVal
                }
            }
            return tensor(newV, [this.shape[1]])
        } else if (axis === 1) {
            if (this.rank > 2) throw new Error('Rank too high for row max, rank is >2')
            let newV = new Array(this.shape[0]).fill(-Infinity)
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    let currVal = this.values[i * this.shape[1] + j]
                    let oldMax = newV[i]
                    if (currVal > oldMax) newV[i] = currVal
                }
            }
            return tensor(newV, [this.shape[0], 1])
        }
    }

    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin(axis?: 0 | 1) {
        if (axis === undefined) {
            let currVal = this.values[0]
            let min = currVal
            for (let i = 0; i < this.values.length; i++) {
                currVal = this.values[i]
                if (currVal < min) min = currVal
            }
            return min
        }
        let newV = new Array(this.shape[1]).fill(Infinity)
        if (axis === 0) {
            if (this.rank > 2) throw new Error('Rank too high for column max, rank is >2')
            for (let i = 0; i < this.shape[1]; i++) {
                for (let j = 0; j < this.shape[0]; j++) {
                    let currVal = this.values[i + this.shape[1] * j]
                    let oldMin = newV[i]
                    if (currVal < oldMin) newV[i] = currVal
                }
            }
            return tensor(newV, [this.shape[1]])
        } else if (axis === 1) {
            if (this.rank > 2) throw new Error('Rank too high for row max, rank is >2')
            let newV = new Array(this.shape[0]).fill(Infinity)
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    let currVal = this.values[i * this.shape[1] + j]
                    let oldMin = newV[i]
                    if (currVal < oldMin) newV[i] = currVal
                }
            }
            return tensor(newV, [this.shape[0], 1])
        }
    }

    argmax() {
        let maxI = 0
        if (this.rank === 0)
            return 0
        else if (this.rank === 1 || this.shape[0] === 1) {
            for (let i = 0; i < this.values.length; i++) {
                if (this.values[i] > this.values[maxI])
                    maxI = i
            }
        } else
            throw new Error("Doesn't handle rank > 1")

        return maxI
    }

    argmin() {
        let minI = 0
        if (this.rank === 0)
            return 0
        else if (this.rank === 1 || this.shape[0] === 1) {
            for (let i = 0; i < this.values.length; i++) {
                if (this.values[i] < this.values[minI])
                    minI = i
            }
        } else
            throw new Error("Doesn't handle rank > 1")

        return minI
    }
}

const doOp = (first: number, second: number, op: string) => {
    if (op === 'add')
        return first + second
    else if (op === 'sub')
        return first - second
    else if (op === 'mul')
        return first * second
    else if (op === 'div')
        return first / second
    else if (op === 'mod')
        return first % second
    else
        throw new Error("Invalid operation code passed")
}

const elementwiseOp = (m: Tensor, n: number | Tensor, op: string, axis?) => {
    let newV = toArr(m.values)
    if (typeof n === 'number') {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n, op)
        }
    } else if (n.rank === 0) {
        let scalarValue = n.values[0]
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], scalarValue, op)
        }
    } else if (axis === 1) {
        if ((n.rank === 1 && n.shape[0] !== m.shape[1]) || (n.shape[0] === 1 && n.shape[1] !== m.shape[1]))
            throw new Error(`Second tensor of shape ${n.shape} should equal first tensor shape on axis=1 but is ${m.shape[1]}`)
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n.values[i % n.values.length], op)
        }
        // } else if (axis === 0) {
        //     for (let i = 0; i < newV.length; i++) {
        //         newV[i] = doOp(newV[i], n.values[i], op)
        //     }
    } else {
        if (m.values.length !== n.values.length)
            throw new Error("Tensors can't be of different sizes for elementwise operation")
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n.values[i], op)
        }
    }
    return tensor(newV, m.shape)
}


