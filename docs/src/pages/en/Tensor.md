---
title: Tensors
description: Tensor information
layout: ../../layouts/MainLayout.astro
---

Tensors are the foundation of this library, they are an n-dimensional matrix which is useful
for representing structures used in neural networks.

## Creating Tensors

Tensors are created with the funtion `tensor()`

```ts
// Pass a nested array
nv.tensor([
  [1, 2],
  [3, 4],
]);
// Or pass a flat array and a shape
nv.tensor([1, 2, 3, 4], [2, 2]);
```

Scalars are represented as `rank = 0` tensors and can be created with

```ts
nv.tensor(3.14);
// or
nv.scalar(3.14);
```

## Fun Facts

All operations return a new Tensor object instead of mutating the original e.g.

```ts
let tensor = nv.tensor([
  [1, 2],
  [3, 4],
]);

console.log(await(await tensor.mul(2)).values());
// [
//  [2, 4],
//  [6, 8],
// ]

console.log(await tensor.values());
// [
//  [1,2],
//  [3,4]
// ]
```

## Notes on Internal Representations

In this library tensors are represented by the Tensor class which contains the properties：

```ts
// The first 4 values are the shape left padded with 0's,
// remaining values are flat values of tensor.
Tensor {
  data: Float32Array;
}
```

Tensor values are stored in row-major order. The shape is represented by 4
values in order of most to least significant dimensions e.g.

```ts
let m = nv.tensor([
  [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
  ],
  [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
  ],
]); // m.data = Float32Array([0, 3, 4, 5, 1, 1, 1, 1, 1, 1, ...])

console.log(await m.shape()); // [3, 4, 5]
console.log(await m.rank()); // 3
```
