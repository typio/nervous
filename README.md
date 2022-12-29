# nervous
> a minimal, functional, ML framework

[Docs](https://nervous-docs.vercel.app) • [Demos](https://nervous-demos.vercel.app)

## Overview

A take what you need ML framework for the browser. Providing [example models](https://nervous-demos.vercel.app) and the components to easily create your own.

Minimal and comprehensible functional ML library, with GPU acceleration using WebGPU *<sup>WebGPU is currently available behind a flag on [Chrome Canary](https://www.google.com/chrome/canary/) and [Firefox Nightly (untested)](https://www.mozilla.org/en-US/firefox/channel/desktop/)</sup>*.

<h2 style="display:inline; margin:0 1rem 1rem 0;">Usage <img width="45" alt="NV sign" style="vertical-align:middle" src="https://user-images.githubusercontent.com/26017543/209094491-6dc7f5aa-4a29-4b89-a06c-969455bbceb5.png"></h2>


``` typescript
import nv from "@typio/nervous"

const main = async () => {
    await nv.init({ backend: 'webgpu' }) 
    let tensor1 = await nv.randomNormal([1024, 1024]) 
    let tensor2 = await nv.randomNormal([1024, 1024])
    let tensorProduct = await tensor1.matmul(tensor2) 
    console.log(tensorProduct.values())
}

main()
```