import * as nv from "nervous";

let tests = [
    {
        suite: "Tensor Constructors",
        tests: [
            {
                name: "scalar()",
                code: async () => {
                    let s = await nv.scalar(4);
                    return [await s.values(), await s.rank(), await s.shape];
                },
                expects: () => [4, 0, [1]],
            },
            {
                name: "tensor()",
                code: async () => {
                    let results = [];

                    let vector = nv.tensor([1, 2, 3]);
                    results.push(await vector.values());
                    results.push(await vector.rank());
                    results.push(await vector.shape);

                    let tensor = nv.tensor([
                        [
                            [
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                            ],
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
                            [
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                            ],
                        ],
                    ]);
                    results.push(await tensor.rank());
                    results.push(await tensor.shape);

                    tensor = nv.tensor(
                        [
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                        ],
                        [2, 2, 3, 4]
                    );
                    results.push(await tensor.rank());
                    results.push(await tensor.shape);

                    return results;
                },
                expects: () => [
                    [1, 2, 3],
                    1,
                    [3],
                    4,
                    [2, 2, 2, 8],
                    4,
                    [2, 2, 3, 4],
                ],
            },
            {
                name: "random()",
                code: async () => {
                    let results = [];
                    let t = await nv.random([4, 3, 2, 5]);
                    results.push(await t.rank());
                    results.push(await t.shape);
                    return results;
                },
                expects: async () => {
                    return [4, [4, 3, 2, 5]];
                },
            },
            {
                name: "fill()",
                code: async () => {
                    let results = [];
                    let t = await nv.fill([2, 3, 4], 3);
                    results.push(await t.rank());
                    results.push(await t.shape);
                    results.push(await t.values());
                    return results;
                },
                expects: async () => {
                    return [
                        3,
                        [2, 3, 4],
                        [
                            [
                                [3, 3, 3, 3],
                                [3, 3, 3, 3],
                                [3, 3, 3, 3],
                            ],
                            [
                                [3, 3, 3, 3],
                                [3, 3, 3, 3],
                                [3, 3, 3, 3],
                            ],
                        ],
                    ];
                },
            },
            {
                name: "ones()",
                code: async () => {
                    let results = [];
                    let t = await nv.ones([2, 3, 4]);
                    results.push(await t.rank());
                    results.push(await t.shape);
                    results.push(await t.values());
                    return results;
                },
                expects: async () => {
                    return [
                        3,
                        [2, 3, 4],
                        [
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ],
                            [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ],
                        ],
                    ];
                },
            },
            {
                name: "zeros()",
                code: async () => {
                    let results = [];
                    let t = await nv.zeros([2, 3, 4]);
                    results.push(await t.rank());
                    results.push(await t.shape);
                    return results;
                },
                expects: async () => {
                    return [
                        3,
                        [2, 3, 4],
                        [
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            [
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                        ],
                    ];
                },
            },
            {
                name: "diag()",
                code: async () => {
                    let results = [];
                    let t = await nv.diag([4, 3, 2, 5]);
                    results.push(await t.values());
                    return results;
                },
                expects: async () => {
                    return [
                        [
                            [4, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 5],
                        ],
                    ];
                },
            },
            {
                name: "eye()",
                code: async () => {
                    let results = [];
                    let t = await nv.eye([3, 3]);
                    results.push(await t.rank());
                    results.push(await t.shape);
                    results.push(await t.values());
                    return results;
                },
                expects: async () => {
                    return [
                        2,
                        [3, 3],
                        [
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                        ],
                    ];
                },
            },
        ],
    },
    {
        suite: "Unary Ops",
        tests: [{
            name: "softmax()",
            code: async () => {
                let results = [];
                results.push(
                    await (
                        await nv
                            .tensor([
                                [0.7705, 10, 0.7223, 0.6692, 0.6393, 0.3074],
                                [0.8244, 0.7386, 0.9349, 0.14, 0.5285, 0.347],
                                [0.3912, 0.9231, 0.7564, 5, 0.1069, 0.1061],
                                [0.0137, 0.4552, 0.3269, 0.3487, 0.7259, 0.2248],
                                [0.5707, 0.8687, 1000, 0.3989, 0.8752, 0.532],
                            ])
                            .softmax(0)
                    ).values(3)
                );
                results.push(
                    await (
                        await nv
                            .tensor([
                                [0.1287, 0.4939, 0.5098, 0.2415, 0.6209, 0.7721],
                                [0.0871, 0.5332, 0.3772, 0.6079, 0.5549, 0.8888],
                                [0.8369, 0.9244, 0.2351, 0.5985, 0.7353, 0.5346],
                            ])
                            .softmax(1)
                    ).values(3)
                );

                return results;
            },

            expects: async () => [
                await nv
                    .tensor([
                        [2.4828e-1, 9.9961e-1, 0.0, 1.2645e-2, 2.0666e-1, 1.9878e-1],
                        [2.6203e-1, 9.4985e-5, 0.0, 7.4488e-3, 1.8499e-1, 2.0681e-1],
                        [1.6991e-1, 1.1423e-4, 0.0, 9.6108e-1, 1.2135e-1, 1.6254e-1],
                        [1.1648e-1, 7.1545e-5, 0.0, 9.1775e-3, 2.2536e-1, 1.8302e-1],
                        [2.0331e-1, 1.0818e-4, 1.0, 9.65e-3, 2.6164e-1, 2.4884e-1],
                    ])
                    .values(3),
                await nv
                    .tensor([
                        [0.1168, 0.1683, 0.171, 0.1307, 0.1911, 0.2222],
                        [0.1063, 0.166, 0.1421, 0.1789, 0.1697, 0.237],
                        [0.1972, 0.2153, 0.1081, 0.1554, 0.1782, 0.1458],
                    ])
                    .values(3),
            ],
        },
        {
            name: "relu()",
            code: async () => {
                let results = [];
                let a
                a = nv.tensor([
                    [0, -1, 2, 4, -19, 9],
                    [4, 3, -5, 4, 19, 1],
                    [0, 8, -2, 2, 19, 9],
                ]);
                results.push(await (a.relu().values()))

                return results
            },

            expects: async () => [
                await nv
                    .tensor([
                        [0, 0, 2, 4, 0, 9],
                        [4, 3, 0, 4, 19, 1],
                        [0, 8, 0, 2, 19, 9],
                    ])
                    .values(),
            ],
        },
        ],
    },
    {
        suite: "Binary Ops",
        tests: [
            {
                name: "add()",
                code: async () => {
                    let results = [];
                    let a, b;

                    a = nv.scalar(3);
                    b = nv.tensor([
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]);
                    results.push(await (await a.add(b)).values());
                    a = nv.tensor([1, 2]);
                    b = nv.tensor([4, 4]);
                    results.push(await (await a.add(b)).values());

                    a = nv.tensor([
                        [0, 0],
                        [0, 0],
                    ]);
                    b = nv.tensor([2, 4]);
                    results.push(await (await a.add(b)).values());

                    a = nv.tensor([
                        [0, 0],
                        [0, 0],
                    ]);
                    b = nv.tensor([[2], [4]]);
                    results.push(await (await a.add(b)).values());

                    a = nv.tensor([
                        [1, 2],
                        [3, 4],
                    ]);
                    b = nv.tensor([
                        [5, 2],
                        [10, 4],
                    ]);
                    results.push(await (await a.add(b)).values());

                    a = nv.tensor([
                        [
                            [
                                [0, 1, 2],
                                [3, 4, 5],
                            ],

                            [
                                [6, 7, 8],
                                [9, 10, 11],
                            ],
                        ],
                        [
                            [
                                [12, 13, 14],
                                [15, 16, 17],
                            ],

                            [
                                [18, 19, 20],
                                [21, 22, 23],
                            ],
                        ],
                    ]);
                    b = nv.tensor([
                        [
                            [
                                [69, 70, 71],
                                [72, 73, 74],
                            ],

                            [
                                [75, 76, 77],
                                [78, 79, 80],
                            ],
                        ],

                        [
                            [
                                [81, 82, 83],
                                [84, 85, 86],
                            ],

                            [
                                [87, 88, 89],
                                [90, 91, 92],
                            ],
                        ],
                    ]);
                    results.push(await (await a.add(b)).values());
                    return results;
                },

                expects: async () => [
                    await nv
                        .tensor([
                            [4, 5, 6],
                            [7, 8, 9],
                            [10, 11, 12],
                        ])
                        .values(),
                    await nv.tensor([5, 6]).values(),
                    await nv
                        .tensor([
                            [2, 4],
                            [2, 4],
                        ])
                        .values(),
                    await nv
                        .tensor([
                            [2, 2],
                            [4, 4],
                        ])
                        .values(),
                    await nv
                        .tensor([
                            [6, 4],
                            [13, 8],
                        ])
                        .values(),
                    await nv
                        .tensor([
                            [
                                [
                                    [69, 71, 73],
                                    [75, 77, 79],
                                ],

                                [
                                    [81, 83, 85],
                                    [87, 89, 91],
                                ],
                            ],

                            [
                                [
                                    [93, 95, 97],
                                    [99, 101, 103],
                                ],

                                [
                                    [105, 107, 109],
                                    [111, 113, 115],
                                ],
                            ],
                        ])
                        .values(),
                ],
            },
            {
                name: "sum()",
                code: async () => {
                    let results = [];
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2],
                                    [4, 5],
                                    [7, 8],
                                ])
                                .sum()
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2],
                                    [4, 5],
                                    [7, 8],
                                ])
                                .sum(0)
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2, 3],
                                    [4, 5, 6],
                                ])
                                .sum(1)
                        ).values()
                    );

                    return results;
                },

                expects: async () => [
                    await nv.scalar(27).values(),
                    await nv.tensor([12, 15], [1, 2]).values(),
                    await nv.tensor([[6], [15]]).values(),
                ],
            },
            {
                name: "argmax()",
                code: async () => {
                    let results = [];
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2],
                                    [4, 5],
                                    [7, 8],
                                ])
                                .argmax()
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 20, 2],
                                    [8, 5, -1],
                                    [7, 8, -10],
                                ])
                                .argmax(0)
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2, 9],
                                    [8, 5, 6],
                                ])
                                .argmax(1)
                        ).values()
                    );

                    return results;
                },

                expects: async () => [
                    await nv.scalar(5).values(),
                    await nv.tensor([1, 0, 0], [1, 3]).values(),
                    await nv.tensor([[2], [0]]).values(),
                ],
            },
        ],
    },
    {
        suite: "Matrix OPs",
        tests: [
            {
                name: "dot()",
                code: async () => {
                    let results = [];

                    let m, n;
                    m = nv.tensor([
                        [4, 1],
                        [2, 2],
                    ]);
                    n = nv.tensor([
                        [5, 3],
                        [27, 9],
                    ]);

                    results.push(await (await m.dot(n)).values());

                    results.push(
                        await (
                            await nv.tensor([10, 20, 30]).dot(nv.tensor([[1], [2], [3]]))
                        ).values()
                    );

                    results.push(
                        await (
                            await nv.tensor([10, 20, 30]).dot(
                                nv.tensor([
                                    [1, 2, 3],
                                    [4, 5, 6],
                                    [4, 5, 6],
                                ])
                            )
                        ).values()
                    );

                    return results;
                },
                expects: async () => {
                    return [
                        [
                            [47, 21],
                            [64, 24],
                        ],
                        140,
                        [210, 270, 330],
                    ];
                },
            },
            {
                name: "tranpose()",
                code: async () => {
                    let results = [];
                    results.push(await (await nv.scalar(3).transpose()).values());
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9],
                                ])
                                .transpose()
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2],
                                    [4, 5],
                                    [7, 8],
                                ])
                                .transpose()
                        ).values()
                    );
                    results.push(
                        await (
                            await nv
                                .tensor([
                                    [1, 2, 3],
                                    [4, 5, 6],
                                ])
                                .transpose()
                        ).values()
                    );

                    return results;
                },

                expects: async () => [
                    await nv.scalar(3).values(),
                    await nv
                        .tensor([
                            [1, 4, 7],
                            [2, 5, 8],
                            [3, 6, 9],
                        ])
                        .values(),
                    await nv
                        .tensor([
                            [1, 4, 7],
                            [2, 5, 8],
                        ])
                        .values(),
                    await nv
                        .tensor([
                            [1, 4],
                            [2, 5],
                            [3, 6],
                        ])
                        .values(),
                ],
            }
        ],
    },
];

export default tests;

// test('reshape', async () => {
//     let tensor = nv.tensor([1, 2, 3, 4, 5, 6], [3, 2])

//     assert.equal(tensor.reshape([2, 3]).getValues(), [[1, 2, 3], [4, 5, 6]])
//     assert.equal(tensor.reshape([2, 3]).shape, [2, 3])
// })

// test('transpose', async () => {
//     let tensor = nv.tensor(4)
//     assert.equal(tensor.transpose().getValues(), 4)

//     tensor = nv.tensor([1, 2, 3, 4])
//     assert.equal(tensor.transpose().getValues(), [[1], [2], [3], [4]])

//     tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.transpose().getValues(), [[1, 3], [2, 4]])

//     tensor = nv.tensor([[1, 2, 3], [4, 5, 6]])
//     assert.equal(tensor.transpose().getValues(), [[1, 4], [2, 5], [3, 6]])

//     // scalar transpose is itself
//     let tensor2 = nv.scalar(4)
//     assert.equal(tensor2.transpose(), tensor2)
// })

// test('mul', async () => {
//     // scalar on nd tensor
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.mul(2).getValues(), [[2, 4], [6, 8]])

//     // tensor scalar on nd tensor
//     let scalar = nv.scalar(2)
//     assert.equal(tensor.mul(scalar).getValues(), [[2, 4], [6, 8]])

//     // 1d tensor on 1d tensor
//     assert.equal(nv.tensor([10, 20, 30]).mul(nv.tensor([1, 2, 3])).getValues(), [10, 40, 90])
// })

// test('div', async () => {
//     // scalar on nd tensor
//     let tensor = nv.tensor([[4, 8], [12, 16]])
//     assert.equal(tensor.div(2).getValues(), [[2, 4], [6, 8]])

//     // tensor scalar on nd tensor
//     let scalar = nv.scalar(2)
//     assert.equal(tensor.div(scalar).getValues(), [[2, 4], [6, 8]])

//     // 1d tensor on 1d tensor
//     assert.equal(nv.tensor([10, 22, 36]).div(nv.tensor([1, 2, 3])).getValues(), [10, 11, 12])
// })

// test('add', async () => {
//     // scalar on nd tensor
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.add(2).getValues(), [[3, 4], [5, 6]])

//     // nd tensor on nd tensor
//     tensor = nv.tensor([[1, 2], [3, 4]])
//     let tensor2 = nv.tensor([[2, 3], [4, 5]])
//     assert.equal(tensor.add(tensor2).getValues(), [[3, 5], [7, 9]])
// })

// test('minus', async () => {
//     // scalar on nd tensor
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.minus(2).getValues(), [[-1, 0], [1, 2]])

//     // nd tensor on nd tensor
//     tensor = nv.tensor([[1, 2], [3, 4]])
//     let tensor2 = nv.tensor([[42, 55], [2, -100]])
//     assert.equal(tensor.minus(tensor2).getValues(), [[-41, -53], [1, 104]])

//     // 2d minus 1d
//     tensor = nv.tensor([[1, 2, 5],
//     [3, 4, 6]])
//     tensor2 = nv.tensor([1, 3, 5])
//     assert.equal(tensor.minus(tensor2, 1).getValues(), [[0, -1, 0], [2, 1, 1]])

//     // // 1d col tensor on nd tensor
//     // tensor = nv.tensor([[1, 2], [3, 4]])
//     // assert.equal(tensor.minus(tensor2).getValues(), [[-41, -53], [1, 104]])
// })

// test('exp', async () => {
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(Math.floor(tensor.exp().sum().getValues()), 84)
//     assert.equal(tensor.exp(2).getValues(), [[2, 4], [8, 16]])
// })

// test('pow', async () => {
//     // TODO:
// })

// test('sum', async () => {
//     let tensor = nv.tensor([[1, 2, 5], [3, 4, 6]])
//     assert.equal(tensor.sum().getValues(), 21)
//     assert.equal(tensor.sum(0).getValues(), [4, 6, 11])
//     assert.equal(tensor.sum(1).getValues(), [[8], [13]])

//     let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
//     assert.equal(tensor2.sum().getValues(), 21)
//     assert.equal(tensor2.sum(0).getValues(), [9, 12])
//     assert.equal(tensor2.sum(1).getValues(), [[3], [7], [11]])
// })

// test('trace', async () => {
//     let tensor = nv.tensor([[1, 2, 5], [3, 4, 6], [2, 5, 23]])
//     assert.equal(tensor.trace(), 28)

//     let scalar = nv.scalar(2)
//     assert.equal(scalar.getValues(), scalar.trace())
// })

// test('fnorm_from_trace', async () => {
//     let tensor = nv.tensor([[1, 2, 5, 5123], [3, 4, 6, 2145], [2, 5, 23, 6661], [4555, 123.23, 12312, 12345]])
//     assert.equal(Math.round(tensor.fNorm()), Math.round(Math.sqrt(tensor.dot(tensor.transpose()).trace())))
// })

// test('trace_invariant_to_transpose', async () => {
//     let tensor = nv.tensor([[1, 2, 5, 5123], [3, 4, 6, 2145], [2, 5, 23, 6661], [4555, 123.23, 12312, 12345]])
//     assert.equal(Math.round(tensor.trace()), Math.round(tensor.transpose().trace()))
// })

// test('trace_and_product_invarience', async () => {
//     let tensor1 = nv.tensor([[65, 76, 14], [6, 98, 69], [44, 22, 56]])
//     let tensor2 = nv.tensor([[79, 22, 93], [29, 57, 60], [63, 23, 27]])
//     let tensor3 = nv.tensor([[20, 96, 22], [95, 26, 3], [4, 49, 32]])
//     assert.equal(tensor1.dot(tensor2).dot(tensor3).trace(), tensor3.dot(tensor1).dot(tensor2).trace())
//     assert.equal(tensor1.dot(tensor2).dot(tensor3).trace(), tensor2.dot(tensor3).dot(tensor1).trace())
// })

// test('applyMax', async () => {
//     let tensor = nv.tensor([[-12, -92], [1234, -123]])
//     assert.equal(tensor.applyMax(0).getFlatValues(), [0, 0, 1234, 0])
// })

// test('applyMin', async () => {
//     let tensor = nv.tensor([[12, 92], [-1234, 123]])
//     assert.equal(tensor.applyMin(0).getFlatValues(), [0, 0, -1234, 0])
// })

// test('getmax', async () => {
//     let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
//     assert.equal(tensor.getmax(), 1234)
//     assert.equal((tensor.getmax(0) as nv.Tensor).getValues(), [1234, 92, 12])
//     assert.equal((tensor.getmax(1) as nv.Tensor).getValues(), [[92], [1234]])

//     let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
//     assert.equal(tensor2.getmax(), 6)
//     assert.equal((tensor2.getmax(0) as nv.Tensor).getValues(), [5, 6])
//     assert.equal((tensor2.getmax(1) as nv.Tensor).getValues(), [[2], [4], [6]])
// })

// test('getmin', async () => {
//     let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
//     assert.equal(tensor.getmin(), -123)
//     assert.equal((tensor.getmin(0) as nv.Tensor).getValues(), [-12, -123, 3])
//     assert.equal((tensor.getmin(1) as nv.Tensor).getValues(), [[-12], [-123]])

//     let tensor2 = nv.tensor([[1, 2], [3, 4], [5, 6]])
//     assert.equal(tensor2.getmin(), 1)
//     assert.equal((tensor2.getmin(0) as nv.Tensor).getValues(), [1, 2])
//     assert.equal((tensor2.getmin(1) as nv.Tensor).getValues(), [[1], [3], [5]])
// })

// test('argmin', async () => {
//     assert.equal(nv.tensor([0, -1, 2, 3]).argmin(), 1)
// })

// test('argmax', async () => {
//     assert.equal(nv.tensor([0, -1, 2, 3]).argmax(), 3)
// })

// test('log', async () => {
//     // TODO:
// })

// test('mean', async () => {
//     // TODO:
// })

// test('sigmoid', async () => {
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.sigmoid().getFlatValues().map(n => Math.round(n * 10E3) / 10E3),
//         [0.7311, 0.8808, 0.9526, 0.9820])
// })

// test('softplus', async () => {
//     let tensor = nv.tensor([[1, 2], [3, 4]])
//     assert.equal(tensor.softplus().getFlatValues().map(n => Math.round(n * 1E4) / 1E4),
//         [1.3133, 2.1269, 3.0486, 4.0181])
// })

// test('reLU', async () => {
//     let tensor = nv.tensor([[-12, 92, 12], [1234, -123, 3]])
//     assert.equal(tensor.reLU().getValues(), [[0, 92, 12], [1234, 0, 3]])
// })

// test('softmax', async () => {
//     let tensor = nv.tensor([-1., 2., -10., 4., 2.9412])
//     assert.equal(tensor.softmax().getFlatValues(4),
//         [0.0045, 0.0909, 0.0000, 0.6716, 0.2330])

//     assert.equal(
//         nv.tensor([
//             [2, 2, 8],
//             [0, 0, 1],
//             [9, 3, 2]
//         ]).softmax().getValues(3),
//         [
//             [0.002, 0.002, 0.995],
//             [0.212, 0.212, 0.576],
//             [0.997, 0.002, 0.001]
//         ])

// });

//test.run()
