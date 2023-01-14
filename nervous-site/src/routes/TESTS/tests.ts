import nv from "nervous";

let tests = [
  {
    suite: "Tensor Constructors",
    tests: [
      {
        name: "scalar()",
        code: async () => {
          let s = await nv.scalar(4);
          return [await s.values(), await s.rank(), await s.shape()];
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
          results.push(await vector.shape());

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
          results.push(await tensor.shape());

          tensor = nv.tensor(
            [
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            ],
            [2, 2, 3, 4]
          );
          results.push(await tensor.rank());
          results.push(await tensor.shape());

          let tensorFromFloatArray = nv.tensor(
            new Float32Array([0, 0, 3, 2, 1, 2, 3, 4, 5, 6])
          );
          results.push(await tensorFromFloatArray.values());

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
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
        ],
      },
      {
        name: "eye()",
        code: async () => {
          let results = [];
          let t = await nv.eye([3, 3]);
          results.push(await t.rank());
          results.push(await t.shape());
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
      {
        name: "fill()",
        code: async () => {
          let results = [];
          let t = await nv.fill([2, 3, 4], 3);
          results.push(await t.rank());
          results.push(await t.shape());
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
          results.push(await t.shape());
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
          results.push(await t.shape());
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
        name: "random()",
        code: async () => {
          // TODO: think of better tasks
          let results = [];
          let t = await nv.random([4, 3, 2, 5]);
          results.push(await t.rank());
          results.push(await t.shape());
          return results;
        },
        expects: async () => {
          return [4, [4, 3, 2, 5]];
        },
      },
    ],
  },
  {
    suite: "Tensor OPs",
    tests: [
      {
        name: "add() on two 1d tensors",
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

          // await new Promise(r => setTimeout(r, 20000));
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
    ],
  },
  {
    suite: "Matrix Multiplication",
    tests: [
      {
        name: "matmul()",
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

          results.push(await (await m.matmul(n)).values());

          results.push(
            await (
              await nv.tensor([10, 20, 30]).matmul(nv.tensor([[1], [2], [3]]))
            ).values()
          );

          results.push(
            await (
              await nv.tensor([10, 20, 30]).matmul(
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
//     assert.equal(Math.round(tensor.fNorm()), Math.round(Math.sqrt(tensor.matmul(tensor.transpose()).trace())))
// })

// test('trace_invariant_to_transpose', async () => {
//     let tensor = nv.tensor([[1, 2, 5, 5123], [3, 4, 6, 2145], [2, 5, 23, 6661], [4555, 123.23, 12312, 12345]])
//     assert.equal(Math.round(tensor.trace()), Math.round(tensor.transpose().trace()))
// })

// test('trace_and_product_invarience', async () => {
//     let tensor1 = nv.tensor([[65, 76, 14], [6, 98, 69], [44, 22, 56]])
//     let tensor2 = nv.tensor([[79, 22, 93], [29, 57, 60], [63, 23, 27]])
//     let tensor3 = nv.tensor([[20, 96, 22], [95, 26, 3], [4, 49, 32]])
//     assert.equal(tensor1.matmul(tensor2).matmul(tensor3).trace(), tensor3.matmul(tensor1).matmul(tensor2).trace())
//     assert.equal(tensor1.matmul(tensor2).matmul(tensor3).trace(), tensor2.matmul(tensor3).matmul(tensor1).trace())
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
