import { test } from 'uvu'
import * as assert from 'uvu/assert'

import * as nv from '../src/index'

test('scalar', () => {
    let tensor = nv.scalar(4)
    assert.equal(tensor.rank, 0)
    assert.equal(tensor.shape, [1])
})

test('tensor', () => {
    let tensor = nv.tensor([
        [
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
        ],
        [
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
        ],
        [
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
        ],
        [
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                ],
            ],
        ],
    ])
    assert.equal(tensor.rank, 6)
    assert.equal(tensor.shape, [4, 2, 3, 1, 2, 8])

    tensor = nv.tensor(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        ],
        [2, 2, 3, 1, 2, 2]
    )
    assert.equal(tensor.rank, 6)
    assert.equal(tensor.shape, [2, 2, 3, 1, 2, 2])
})

test('eye', () => {
    let tensor = nv.eye([3, 3])
    assert.equal(tensor.rank, 2)
    assert.equal(tensor.shape, [3, 3])
    assert.equal(tensor.getValues(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
})

test('zeros', () => {
    let tensor = nv.zeros([2, 3, 4])
    assert.equal(tensor.rank, 3)
    assert.equal(tensor.shape, [2, 3, 4])
})

test('random', () => {
    let tensor = nv.random([4, 3, 2, 5], 0, 10, true)
    assert.equal(tensor.rank, 4)
    assert.equal(tensor.shape, [4, 3, 2, 5])
    assert.equal(tensor.values[0] % 1, 0)

    tensor = nv.random([1, 2, 3, 4, 7], 0, 10, false)
    assert.equal(tensor.rank, 5)
    assert.equal(tensor.shape, [1, 2, 3, 4, 7])
    assert.not.equal(tensor.values[0] % 1, 0)
})

test('diag', () => {
    let tensor = nv.diag([4, 3, 2, 5])
    assert.equal(tensor.getValues(), [[4, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 5]])
})


test('reshape', () => {
    let tensor = nv.tensor([1, 2, 3, 4, 5, 6], [3, 2])

    assert.equal(tensor.reshape([2, 3]).getValues(), [[1, 2, 3], [4, 5, 6]])
    assert.equal(tensor.reshape([2, 3]).shape, [2, 3])
})

test('transpose', () => {
    let tensor = nv.tensor(4)
    assert.equal(tensor.transpose().getValues(), 4)

    tensor = nv.tensor([1, 2, 3, 4])
    assert.equal(tensor.transpose().getValues(), [[1], [2], [3], [4]])

    tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.transpose().getValues(), [[1, 3], [2, 4]])

    tensor = nv.tensor([[1, 2, 3], [4, 5, 6]])
    assert.equal(tensor.transpose().getValues(), [[1, 4], [2, 5], [3, 6]])

    // scalar transpose is itself
    let tensor2 = nv.scalar(4)
    assert.equal(tensor2.transpose(), tensor2)
})

test('matMul', () => {
    // 1d tensor on 1d tensor
    assert.equal(nv.tensor([10, 20, 30]).matMul(nv.tensor([1, 2, 3])).getValues(), 140)

    // 1d tensor on 2d tensor
    // assert.equal(nv.tensor([10, 20, 30]).matMul(nv.tensor([1, 2, 3])).getValues(), 140)

    // 2d tensor on 2d tensor
    assert.equal(
        nv.tensor([[1, 2, 3], [4, 5, 6]]).matMul(nv.tensor([[1, 2], [3, 4], [5, 6]])).getValues(),
        [[22, 28],
        [49, 64]]
    )
})

test('mul', () => {
    // scalar on nd tensor
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.mul(2).getValues(), [[2, 4], [6, 8]])

    // tensor scalar on nd tensor
    let scalar = nv.scalar(2)
    assert.equal(tensor.mul(scalar).getValues(), [[2, 4], [6, 8]])

    // 1d tensor on 1d tensor
    assert.equal(nv.tensor([10, 20, 30]).mul(nv.tensor([1, 2, 3])).getValues(), [10, 40, 90])
})

test('div', () => {
    // scalar on nd tensor
    let tensor = nv.tensor([[4, 8], [12, 16]])
    assert.equal(tensor.div(2).getValues(), [[2, 4], [6, 8]])

    // tensor scalar on nd tensor
    let scalar = nv.scalar(2)
    assert.equal(tensor.div(scalar).getValues(), [[2, 4], [6, 8]])

    // 1d tensor on 1d tensor
    assert.equal(nv.tensor([10, 22, 36]).div(nv.tensor([1, 2, 3])).getValues(), [10, 11, 12])
})

test('add', () => {
    // scalar on nd tensor
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.add(2).getValues(), [[3, 4], [5, 6]])

    // nd tensor on nd tensor
    tensor = nv.tensor([[1, 2], [3, 4]])
    let tensor2 = nv.tensor([[2, 3], [4, 5]])
    assert.equal(tensor.add(tensor2).getValues(), [[3, 5], [7, 9]])
})

test('minus', () => {
    // scalar on nd tensor
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.minus(2).getValues(), [[-1, 0], [1, 2]])

    // nd tensor on nd tensor
    tensor = nv.tensor([[1, 2], [3, 4]])
    let tensor2 = nv.tensor([[42, 55], [2, -100]])
    assert.equal(tensor.minus(tensor2).getValues(), [[-41, -53], [1, 104]])

    // 1d col tensor on nd tensor
    // tensor = nv.tensor([[1, 2, 5],
    //                     [3, 4, 6]])
    // tensor2 = nv.tensor([1, 3, 5])
    // assert.equal(tensor.minus(tensor2).getValues(), [[-41, -53], [1, 104]])

    // // 1d col tensor on nd tensor
    // tensor = nv.tensor([[1, 2], [3, 4]])
    // assert.equal(tensor.minus(tensor2).getValues(), [[-41, -53], [1, 104]])
})

test('exp', () => {
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(Math.floor(tensor.exp().sum().getValues()), 84)
    assert.equal(tensor.exp(2).getValues(), [[2, 4], [8, 16]])
})

test('sum', () => {
    let tensor = nv.tensor([[1, 2, 5], [3, 4, 6]])
    assert.equal(tensor.sum().getValues(), 21)
    assert.equal(tensor.sum(0).getValues(), [4, 6, 11])
    assert.equal(tensor.sum(1).getValues(), [[8], [13]])

    let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
    assert.equal(tensor2.sum().getValues(), 21)
    assert.equal(tensor2.sum(0).getValues(), [9, 12])
    assert.equal(tensor2.sum(1).getValues(), [[3], [7], [11]])
})

test('trace', () => {
    let tensor = nv.tensor([[1, 2, 5], [3, 4, 6], [2, 5, 23]])
    assert.equal(tensor.trace(), 28)

    let scalar = nv.scalar(2)
    assert.equal(scalar.getValues(), scalar.trace())
})

test('fnorm_from_trace', () => {
    let tensor = nv.tensor([[1, 2, 5, 5123], [3, 4, 6, 2145], [2, 5, 23, 6661], [4555, 123.23, 12312, 12345]])
    assert.equal(Math.round(tensor.fNorm()), Math.round(Math.sqrt(tensor.matMul(tensor.transpose()).trace())))
})

test('trace_invariant_to_transpose', () => {
    let tensor = nv.tensor([[1, 2, 5, 5123], [3, 4, 6, 2145], [2, 5, 23, 6661], [4555, 123.23, 12312, 12345]])
    assert.equal(Math.round(tensor.trace()), Math.round(tensor.transpose().trace()))
})

test('trace_and_product_invarience', () => {
    let tensor1 = nv.tensor([[65, 76, 14], [6, 98, 69], [44, 22, 56]])
    let tensor2 = nv.tensor([[79, 22, 93], [29, 57, 60], [63, 23, 27]])
    let tensor3 = nv.tensor([[20, 96, 22], [95, 26, 3], [4, 49, 32]])
    assert.equal(tensor1.matMul(tensor2).matMul(tensor3).trace(), tensor3.matMul(tensor1).matMul(tensor2).trace())
    assert.equal(tensor1.matMul(tensor2).matMul(tensor3).trace(), tensor2.matMul(tensor3).matMul(tensor1).trace())
})

test('applyMax', () => {
    let tensor = nv.tensor([[-12, -92], [1234, -123]])
    assert.equal(tensor.applyMax(0).getFlatValues(), [0, 0, 1234, 0])
})

test('applyMin', () => {
    let tensor = nv.tensor([[12, 92], [-1234, 123]])
    assert.equal(tensor.applyMin(0).getFlatValues(), [0, 0, -1234, 0])
})

test('getMax', () => {
    let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
    assert.equal(tensor.getMax(), 1234)
    assert.equal((tensor.getMax(0) as nv.Tensor).getValues(), [1234, 92, 12])
    assert.equal((tensor.getMax(1) as nv.Tensor).getValues(), [[92], [1234]])

    let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
    assert.equal(tensor2.getMax(), 6)
    assert.equal((tensor2.getMax(0) as nv.Tensor).getValues(), [5, 6])
    assert.equal((tensor2.getMax(1) as nv.Tensor).getValues(), [[2], [4], [6]])
})

test('getMin', () => {
    let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
    assert.equal(tensor.getMin(), -123)
    assert.equal((tensor.getMin(0) as nv.Tensor).getValues(), [-12, -123, 3])
    assert.equal((tensor.getMin(1) as nv.Tensor).getValues(), [[-12], [-123]])

    let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
    assert.equal(tensor2.getMin(), 1)
    assert.equal((tensor2.getMin(0) as nv.Tensor).getValues(), [1, 2])
    assert.equal((tensor2.getMin(1) as nv.Tensor).getValues(), [[1], [3], [5]])
})

test('sigmoid', () => {
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.sigmoid().getFlatValues().map(n => Math.round(n * 10E3) / 10E3),
        [0.7311, 0.8808, 0.9526, 0.9820])
})

test('softplus', () => {
    let tensor = nv.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.softplus().getFlatValues().map(n => Math.round(n * 10E3) / 10E3),
        [1.3133, 2.1269, 3.0486, 4.0181])
})

test('relu', () => {
    let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
    assert.equal(tensor.relu().getValues(), [[0, 92, 12], [1234, 0, 3]])
})

test('softmax', () => {
    let tensor = nv.tensor([-1., 2., -10., 4., 2.9412])
    assert.equal(tensor.softmax().getFlatValues().map(n => Math.round(n * 10E3) / 10E3),
        [0.0045, 0.0909, 0.0000, 0.6716, 0.2330])
})

test.run()