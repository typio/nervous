import { test } from 'uvu'
import * as assert from 'uvu/assert'

import * as lt from '../src/tensor'

test('scalar', () => {
    let tensor = lt.scalar(4)
    assert.equal(tensor.rank, 0)
    assert.equal(tensor.shape, [1])
})

test('tensor', () => {
    let tensor = lt.tensor([
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

    tensor = lt.tensor(
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
    let tensor = lt.eye([3, 3])
    assert.equal(tensor.rank, 2)
    assert.equal(tensor.shape, [3, 3])
    assert.equal(tensor.getValues(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
})

test('zeroes', () => {
    let tensor = lt.zeroes([2, 3, 4])
    assert.equal(tensor.rank, 3)
    assert.equal(tensor.shape, [2, 3, 4])
})

test('random', () => {
    let tensor = lt.random([4, 3, 2, 5], 0, 10, true)
    assert.equal(tensor.rank, 4)
    assert.equal(tensor.shape, [4, 3, 2, 5])
    assert.equal(tensor.values[0] % 1, 0)

    tensor = lt.random([1, 2, 3, 4, 7], 0, 10, false)
    assert.equal(tensor.rank, 5)
    assert.equal(tensor.shape, [1, 2, 3, 4, 7])
    assert.not.equal(tensor.values[0] % 1, 0)
})


test('reshape', () => {
    let tensor = lt.tensor([1, 2, 3, 4, 5, 6], [3, 2])

    assert.equal(tensor.reshape([2, 3]).getValues(), [[1, 2, 3], [4, 5, 6]])
    assert.equal(tensor.reshape([2, 3]).shape, [2, 3])
})

test('transpose', () => {
    let tensor = lt.tensor(4)
    assert.equal(tensor.transpose().getValues(), 4)

    tensor = lt.tensor([1, 2, 3, 4])
    assert.equal(tensor.transpose().getValues(), [[1], [2], [3], [4]])

    tensor = lt.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.transpose().getValues(), [[1, 3], [2, 4]])

    tensor = lt.tensor([[1, 2, 3], [4, 5, 6]])
    assert.equal(tensor.transpose().getValues(), [[1, 4], [2, 5], [3, 6]])
})

test('mul', () => {
    // scalar on nd tensor
    let tensor = lt.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.mul(2).getValues(), [[2, 4], [6, 8]])

    // 1d tensor on 1d tensor
    assert.equal(lt.tensor([10, 20, 30]).mul(lt.tensor([1, 2, 3])).getValues(), [10, 40, 90])
})

// test('exp', () => {
//     let tensor = lt.tensor([[1, 2], [3, 4]])
//     assert.equal(Math.floor(tensor.exp().sum()), 70)
//     assert.equal(tensor.exp(2).getValues(), [[1, 4], [9, 16]])
// })

test('sum', () => {
    let tensor = lt.tensor([[1, 2], [3, 4]])
    assert.equal(tensor.sum(), 10)
})



test.run()