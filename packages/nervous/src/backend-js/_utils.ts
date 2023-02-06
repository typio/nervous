import { BinaryOp, Tensor } from '../tensor'
import { toArr } from '../tensorUtils'

export const doOp = (first: number, second: number, op: BinaryOp) => {
    if (op === BinaryOp.add) return first + second
    else if (op === BinaryOp.minus) return first - second
    else if (op === BinaryOp.mul) return first * second
    else if (op === BinaryOp.div) return first / second
    else if (op === BinaryOp.mod) return first % second
    else throw new Error('Invalid operation code passed')
}

export const elementwiseOp = (m: Tensor, n: number | Tensor, op: BinaryOp) => {
    let mShape = m.shape()
    // @ts-ignore
    let nShape = n.constructor === Number ? undefined : n.shape()

    let mFlatValues = m.flatValues()
    // @ts-ignore
    let nFlatValues = n.constructor === Number ? [n] : n.flatValues()

    let newV = mFlatValues

    // @ts-ignore
    if (n.constructor === Number || n.rank() === 0) {
        // @ts-ignore
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[0], op)
        }
        // @ts-ignore
    } else if (mFlatValues.length === nFlatValues.length) {
        // @ts-ignore
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[i], op)
        }
    } else if (nShape[0] === 1 && nShape[1] === mShape[1]) {
        // @ts-ignore
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[i % nFlatValues.length], op)
        }
    } else {
        throw new Error("Couldn't match tensor shapes.")
    }
    // @ts-ignore
    return new Tensor(newV, m.shape())
}

export const broadcast = (a: Tensor, func: any) => {
    let newV = []
    let aFlatValues = a.flatValues()
    // @ts-ignore
    for (let i = 0; i < aFlatValues.length; i++) {
        newV[i] = func(aFlatValues[i])
    }
    return new Tensor(newV, a.shape())
}
