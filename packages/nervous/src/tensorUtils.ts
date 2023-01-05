import { Rank1To4Array } from "./tensor"

/** Convert FloatArray to Array (Array.from() and slice are slow...) */
export const toArr = (floatArr: Float32Array, decimals?: number, startIndex = 0) => {
    let arr = []
    if (decimals === undefined)
        for (let i = startIndex; i < floatArr.length; i++)
            arr[i - startIndex] = floatArr[i]
    else
        for (let i = startIndex; i < floatArr.length; i++)
            arr[i - startIndex] = Number(floatArr[i].toFixed(decimals))
    return arr
}

export const calcShape = (values: Rank1To4Array): number[] => {
    // TODO: check for and warn if tensor is jagged
    let shape: number[] = []
    let subValues: Rank1To4Array | number = values
    while (subValues.constructor === Array) {
        shape.push(subValues.length)

        subValues = subValues[0]
    }
    return shape
}

export const flatLengthFromShape = (shape: number[]) => {
    if (shape[0] === 0) return 1
    // reduce is fine considering max array length is 6
    return shape.reduce((previousValue, currentValue) => Math.max(1, previousValue) * Math.max(1, currentValue), 1)
}

export const toNested = (values: number[], shape: number[]) => {
    if (flatLengthFromShape(shape) !== values.length)
        throw new Error(`New shape is not compatible with initial values length: shape: ${shape} values.length: ${values.length}.`)

    // if (shape = [0])

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
