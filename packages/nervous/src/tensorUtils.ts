import { Rank1To4Array } from './tensor'

/** Convert FloatArray to Array (Array.from() and slice are slow...) */
export const toArr = (floatArr: Float32Array, decimals?: number, startIndex = 0): number[] => {
  let arr = []
  if (decimals === undefined) for (let i = startIndex; i < floatArr.length; i++) arr[i - startIndex] = floatArr[i]
  else for (let i = startIndex; i < floatArr.length; i++) arr[i - startIndex] = Number(floatArr[i].toFixed(decimals))
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

  if (shape.length === 1) {
    // if vector, set first el of shape to 1 for 1 row count
    shape = [values.length]
  }

  return shape
}

export const flatLengthFromShape = (shape: number[]) => {
  // reduce is fine considering max array length is 4
  return shape.reduce((previousValue, currentValue) => Math.max(1, previousValue) * Math.max(1, currentValue), 1)
}

export const toNested = (values: number[], shape: number[]) => {
  // if (flatLengthFromShape(shape) !== values.length)
  // 	throw new Error(
  // 		`New shape is not compatible with initial values length: shape: ${shape} values.length: ${values.length}.`
  // 	)
  //
  if (shape.length === 1) {
    return values
  }
  let nestedArr = []
  let subArrSize = 1
  for (let i = 1; i < shape.length; i++) {
    subArrSize *= shape[i]
  }
  let subArrStartIndex = 0
  while (subArrStartIndex < values.length) {
    let subArrEndIndex = subArrStartIndex + subArrSize
    nestedArr.push(toNested(values.slice(subArrStartIndex, subArrEndIndex), shape.slice(1)))
    subArrStartIndex = subArrEndIndex
  }
  return nestedArr
}

export const arrMax = (arr: number[]): number => {
  let max = -Infinity
  let c = -Infinity
  for (let i = 0; i < arr.length; i++) {
    c = arr[i]
    if (c > max) max = c
  }
  return max
}

/** formats a shape in the 4 element left padded 0 array form*/
export const padShape = (_shape: number | number[], _pV?: number) => {
  let pV = _pV === undefined ? 0 : _pV
  if (_shape.constructor === Number) return [pV, pV, pV, _shape]
  _shape = _shape as number[]
  if (_shape.length > 4) throw new Error('shape length should be less than or equal to 4')
  let shape = [pV, pV, pV, pV]
  for (let i = 4 - _shape.length; i < 4; i++) shape[i] = _shape[i - 4 + _shape.length]

  return shape
}
