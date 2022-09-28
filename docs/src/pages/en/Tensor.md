---
title: Tensors
description: Tensor information
layout: ../../layouts/MainLayout.astro
---

Tensors are the foundation of this library, they are an n-dimensional matrix which is useful 
for representing structures used in neural networks.

In this library they are represented by the Tensor class which contains the properties： 
```ts
values: Float32Array // flat array of values
shape: number[] // shape of tensor
rank: 1 | 2 |3 | 4 | 5 | 6 // number of dimensions (shape length)
```

## Creating Tensors

Tensors are created with the funtion `tensor()`

```ts
// Pass a nested array
lt.tensor([[1, 2], [3, 4]])
// Or pass a flat array and a shape
lt.tensor([1, 2, 3, 4], [2, 2])
```

Scalars are represented as `rank = 0` tensors and can be created with
```ts
lt.tensor(3.14)
// or
lt.scalar(3.14)
```

## Fun Facts

Our Tensor class is unusual in that all operations return a new Tensor object 
instead of mutating the original e.g.
```ts
let tensor = lt.tensor([[1, 2], [3, 4]])

tensor.mul(2).print()
/* console.log
*  [[2,4],[6,8]]
*/

tensor.print() // orignal tensor is preserved
/* console.log
*  [[1,2],[3,4]]
*/
```