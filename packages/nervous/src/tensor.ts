// TODO: CHANGE new Array()'s to Float32Array's

type Rank1To6Array = number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][]

/** Convert FloatArray to Array (Array.from() is slow...) */
const toArr = (floatArr: Float32Array | Float64Array) => {
    let arr = []
    for (let i = 0; i < floatArr.length; i++)
        arr[i] = floatArr[i]
    return arr
}

const calcShape = (values: Rank1To6Array) => {
    let shape: number[] = []
    let subValues: Rank1To6Array | number = values
    while (subValues.constructor === Array) {
        shape.push(subValues.length)

        subValues = subValues[0]
    }
    return shape
}

export const flatLengthFromShape = (shape: number[]) => {
    // reduce is fine considering max array length is 6
    return shape.reduce((previousValue, currentValue) => previousValue * currentValue, 1)
}

const do_op = (first: number, second: number, op: string) => {
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

const elementwise_op = (m: Tensor, n: number | Tensor, op: string, axis?) => {
    let newV = toArr(m.values)
    if (typeof n === 'number') {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = do_op(newV[i], n, op)
        }
    } else if (n.rank === 0) {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = do_op(newV[i], n.values[0], op)
        }
        // if input is row, op v[i] to each v[i][j]
    } else if (axis === 1 || (axis === undefined && n.shape[0] === 1 && n.shape[1] === m.shape[1])) {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = do_op(newV[i], n.values[i], op)
        }
        // if input is col, op v[i] to each v[i][j]
    } else if (axis === 1 || n.shape[1] === 1) {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = do_op(newV[i], n.values[i], op)
        }
    } else {
        if (m.values.length !== n.values.length)
            throw new Error("Tensors can't be of different sizes for elementwise operation")
        for (let i = 0; i < newV.length; i++) {
            newV[i] = do_op(newV[i], n.values[i], op)
        }
    }
    return tensor(newV, m.shape)
}

const toNested = (values: number[], shape: number[]) => {
    if (flatLengthFromShape(shape) !== values.length)
        throw new Error("New shape is not compatible with initial values length.")

    if (shape.length === 1) {
        return values
    } else if (shape.length === 2) {
        let newV = new Array(shape[0])
        for (let i = 0; i < shape[0]; i++) {
            newV[i] = new Array(shape[1])
            for (let j = 0; j < shape[1]; j++) {
                newV[i][j] = values[i * shape[1] + j]
            }
        }
        return newV
    } else {
        // TODO: try to optimize
        // https://stackoverflow.com/a/69584753/6806458
        let elementI = 0
        const nest = (shapeI: number) => {
            let result: any = []
            if (shapeI === shape.length - 1) {
                // ARMAGEDDON: wtf is this
                result = result.concat(values.slice(elementI, elementI + shape[shapeI]))
                elementI += shape[shapeI]
            } else {
                for (let i = 0; i < shape[shapeI]; i++) {
                    result.push(nest(shapeI + 1)) // NUCLEAR FALLOUT: wat
                }
            }
            return result
        }
        return nest(0) // NUCLEAR WINTER: AHHHH
    }
}

export class Tensor {
    readonly values: Float32Array = new Float32Array(0)
    readonly rank: 0 | 1 | 2 | 3 | 4 | 5 | 6 = 0
    readonly shape: number[] = [0]

    constructor(values: number | Rank1To6Array, shape?: number[]) {
        if (values !== undefined && shape === undefined) {
            if (values.constructor !== Array) {
                // if scalar
                // @ts-ignore: I checked the type
                values = [values] // store scalar number as number[]
                this.shape = [1]
                this.rank = 0
            } else {
                this.shape = calcShape(values)
                this.rank = this.shape.length as typeof this.rank
            }
            // @ts-ignore: I checked the type
            let flatValues = values.flat(5)
            this.values = new Float32Array(flatValues)
        } else if (values !== undefined && shape !== undefined) {
            if (values.constructor === Array && values[0].constructor === Array)
                throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

            // @ts-ignore: I checked the type
            let flatValues: number[] = values

            if (flatLengthFromShape(shape) !== flatValues.length)
                throw new Error("Values don't fit into shape.")

            this.shape = shape
            this.rank = shape.length as typeof this.rank // good ts? 🤔
            this.values = new Float32Array(flatValues)
        }
    }

    /** return nested tensor values */
    getValues() {
        if (this.rank === 0) return this.values[0]
        return toNested(toArr(this.values), this.shape)
    }

    /** return flat tensor values */
    getFlatValues() {
        return toArr(this.values)
    }

    /** console.log nested tensor values */
    print() {
        console.log(JSON.stringify(this.getValues()))
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
    dot(m: Tensor | number) {
        if (typeof m === 'number' || this.rank === 0)
            throw new Error("Please use Tensor.mul() for tensor scalar multiplication.")

        // if 1d * 1d OR 2d (1 row, n cols) * 2d (2 rows, 1 col)
        if ((this.rank === 1 && m.rank === 1) || (this.rank === 2 && m.rank === 2 && this.shape[0] === 1 && m.shape[1] === 1)) {
            let newV: number = 0
            for (let i = 0; i < this.values.length; i++)
                newV += this.values[i] * m.values[i]
            return new Tensor(newV)
        }

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

        throw new Error("Tensor dot on rank > 2 tensors not yet supported.")
    }

    inverse() {
        throw new Error("Not impl., maybe ever")
    }

    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul(m: Tensor | number) {
        return elementwise_op(this, m, 'mul')
    }

    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div(d: Tensor | number) {
        return elementwise_op(this, d, 'div')
    }

    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add(a: number | Tensor) {
        return elementwise_op(this, a, 'add')
    }

    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus(s: number | Tensor) {
        return elementwise_op(this, s, 'sub')
    }

    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod(m: number | Tensor) {
        return elementwise_op(this, m, 'mod')
    }

    /** create tensor with sigmoid done to all values  */
    sigmoid() {
        let newV = []
        for (let i = 0; i < this.values.length; i++) {
            newV[i] = 1 / (1 + Math.E ** -this.values[i])
        }
        return new Tensor(newV, this.shape)
    }

    /** create tensor with softplus done to all values  */
    softplus() {
        let newV = []
        for (let i = 0; i < this.values.length; i++) {
            newV[i] = Math.log(1 + Math.E ** this.values[i])
        }
        return new Tensor(newV, this.shape)
    }

    /** create tensor with relu done to all values  */
    relu() {
        let newV = []
        for (let i = 0; i < this.values.length; i++) {
            let v = this.values[i]
            newV[i] = v > 0 ? v : 0
        }
        return new Tensor(newV, this.shape)
    }

    /** create tensor of exponentials of all values on e, or given base  */
    exp(base?: number) {

        let newV = []
        if (base !== undefined)
            for (let i = 0; i < this.values.length; i++)
                newV[i] = base ** this.values[i]
        else
            for (let i = 0; i < this.values.length; i++)
                newV[i] = Math.E ** this.values[i]
        return new Tensor(newV, this.shape)
    }

    /** return the lp norm, default p is 2  */
    lpnorm(p?: number) {
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
        let newV = []
        for (let i = 0; i < this.values.length; i++)
            newV[i] = (this.values[i] > n) ? this.values[i] : n
        return tensor(newV, this.shape)
    }

    /** returns tensor with elementwise min of old value vs input number */
    applyMin(n: number) {
        let newV = []
        for (let i = 0; i < this.values.length; i++)
            newV[i] = (this.values[i] < n) ? this.values[i] : n
        return tensor(newV, this.shape)
    }

    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getMax(axis?: 0 | 1) {
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
    getMin(axis?: 0 | 1) {
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
}

/**
 * Pass a value
 * ```ts
 * scalar(4)
 * ```
 */
export const scalar = (value: number) => {
    return new Tensor(value)
}

/**
 * Pass a nested array
 * ```ts
 * tensor([[1,2],[3,4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor = (values: number | Rank1To6Array, shape?: number[]) => {
    if (values.constructor === Array && values.length === 1) return new Tensor(values[0])
    return new Tensor(values, shape)
}

/**
 * Pass array of row number and column number, and the position for the one
 * ```ts
 * oneHot([2, 2], [0,1])
 * ```
 * Or flat index
 * ```ts
 * oneHot([2, 2], 1)
 * ```
 */
export const oneHot = (dim: number[] | number, index: number | number[]) => {
    throw new Error('Not implemented.')
}

/**
 * Pass array of row number and column number
 * ```ts
 * eye([2, 2])
 * ```
 * Or a number for both
 * ```ts
 * eye(2); eye([2])
 * ```
 */
export const eye = (dim: number[] | number, offset?: number) => {
    let rowN: number, colN: number
    if (typeof dim === 'number') {
        dim = [dim]
    }
    rowN = dim[0]
    if (dim.length === 1)
        colN = dim[0]
    else
        colN = dim[1]

    let idx = offset ?? 0
    let values = new Array(rowN * colN).fill(0)
    while (idx < rowN * colN) {
        values[idx] = 1
        idx += colN + 1
    }

    return new Tensor(values, [rowN, colN])
}

/**
 * Pass shape of matrix
 * ```ts
 * random([2, 2])
 * ```
 * And optionally min (inclusive), max (exclusive), and integer
 * ```ts
 * random([2, 2], 0, 10, true)
 * ```
 */
export const random = (shape: number[], min?: number, max?: number, integer?: boolean) => {
    if ((min !== undefined && max === undefined) || (max !== undefined && min === undefined))
        throw new Error('Must have either both min and max params or neither.')

    if (min !== undefined && max !== undefined)
        if (integer) {
            if (shape.constructor === Array)
                return new Tensor(
                    Array.from({ length: flatLengthFromShape(shape) }, () => Math.floor(Math.random() * (max - min) + min)),
                    shape
                )
            else
                return new Tensor(
                    // @ts-ignore: I checked the type
                    Array.from({ length: shape }, () => Math.floor(Math.random() * (max - min) + min))
                )
        } else {
            if (shape.constructor === Array)
                return new Tensor(
                    Array.from({ length: flatLengthFromShape(shape) }, () => Math.random() * (max - min) + min),
                    shape)
            else
                return new Tensor(
                    // @ts-ignore: I checked the type
                    Array.from({ length: shape }, () => Math.random() * (max - min) + min))
        }

    if (shape.constructor === Array)
        return new Tensor(
            Array.from({ length: flatLengthFromShape(shape) }, () => Math.random()),
            shape
        )
    else
        return new Tensor(
            // @ts-ignore: I checked the type
            Array.from({ length: shape }, () => Math.random())
        )
}

/**
 * Pass shape of matrix
 * ```ts
 * fill([2, 2], 1)
 * ```
 */
export const fill = (shape: number | number[], value: number) => {
    if (shape.constructor === Array)
        return new Tensor(new Array(flatLengthFromShape(shape)).fill(value), shape)
    else // @ts-ignore
        return new Tensor(new Array(shape).fill(value))
}

export const zeroes = (shape: number | number[]) => {
    return fill(shape, 0)
}

export const ones = (shape: number | number[]) => {
    return fill(shape, 1)
}