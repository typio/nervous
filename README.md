# nervous

>  Web Machine Learning Framework

[Docs](https://nervous-docs.vercel.app) • [Demos](https://nervous-demos.vercel.app)

## Overview

An ML framework for the browser, with these goals:
* simple yet efficient
* less convoluted than TensorFlow and pyTorch
* supporting levels of interaction from tensor operations to full neural nets.

Features GPU acceleration using WebGPU and compute shaders.

I decided to remove the dual-backend system supporting JavaScript and WebGPU and go forward with
just WebGPU. It's a hard choice because I've invested so much in the JS backend, but it will be
useless outside of educational purposes once WebGPU is released and removing the multiple backend
support will reduce the architectural complexity of the library and the lag time of new features.

## Usage

*package is not yet published*
```typescript
import nv from "@typio/nervous";

const main = async () => {
    await nv.init();
    const [n, m] = await Promise.all([
        nv.random([2047, 512]),
        nv.random([512, 2047]),
    ]);
    let res = n.dot(m);

    await res.print(4);
};

main();
```
