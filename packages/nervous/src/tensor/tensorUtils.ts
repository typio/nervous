import { TensorDataValues } from "./tensor"

/** Convert FloatArray to Array (Array.from() and slice are slow...) */
export const toArr = (floatArr: Float32Array, decimals?: number, startIndex = 0): number[] => {
    let arr = []
    if (decimals === undefined) for (let i = startIndex; i < floatArr.length; i++) arr[i - startIndex] = floatArr[i]
    else for (let i = startIndex; i < floatArr.length; i++) arr[i - startIndex] = Number(floatArr[i].toFixed(decimals))
    return arr
}

export const calcShape = (values: TensorDataValues): number[] => {
    if (values.constructor !== Array) return [1]

    let shape: number[] = []
    let subValues: TensorDataValues = values
    while (subValues.constructor === Array) {
        shape.push(subValues.length)

        subValues = subValues[0]
    }

    if (shape.length === 1) {
        // if vector, set first el of shape to 1 for 1 row count
        shape = [values.length]
    }

    return shape
}

export const flatLengthFromShape = (shape: number[]) => {
    // reduce is fine considering max array length is short
    return shape.reduce((previousValue, currentValue) => Math.max(1, previousValue) * Math.max(1, currentValue), 1)
}

export const toNested = (values: number[], _shape: number[]) => {
    let shape = unpadShape(_shape)
    // if (flatLengthFromShape(shape) !== values.length)
    // 	throw new Error(
    // 		`New shape is not compatible with initial values length: shape: ${shape} values.length: ${values.length}.`
    // 	)
    //
    if (shape.length === 1)
        return values

    let nestedArr = []
    let subArrSize = 1
    for (let i = 1; i < shape.length; i++) {
        subArrSize *= shape[i]
    }
    let subArrStartIndex = 0
    while (subArrStartIndex < values.length) {
    if (shape.length === 0) return nestedArr
        let subArrEndIndex = subArrStartIndex + subArrSize
        nestedArr.push(toNested(values.slice(subArrStartIndex, subArrEndIndex), shape.slice(1)))
        subArrStartIndex = subArrEndIndex
    }
    return nestedArr
}

/** formats a shape in a fixed size 7 array, right padded with 0's*/
export const padShape = (_shape: number | number[], _pV?: number) => {
    // pV is padding value
    let pV = _pV === undefined ? 0 : _pV
    if (_shape.constructor === Number)
        return [_shape, pV, pV, pV, pV, pV, pV]
    _shape = _shape as number[]
    if (_shape.length > 7) throw new Error('shape length should be less than or equal to 7')
    let shape = [pV, pV, pV, pV, pV, pV, pV]
    for (let i = 0; i < _shape.length; i++) shape[i] = _shape[i]
    return shape
}

export const unpadShape = (_shape: number[]) => {
    let shape = []
    for (let i = 0; i < _shape.length; i++) {
        if (_shape[i] !== 0) shape.push(_shape[i])
    }
    return shape
}
