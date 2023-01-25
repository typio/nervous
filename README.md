# nervous

> a minimal, functional, ML framework

[Docs](https://nervous-docs.vercel.app) • [Demos](https://nervous-demos.vercel.app)

## Overview

An ML framework for the browser with the goal of being simple yet efficient, and less convoluted than popular production grade frameworks. I provide [example models](https://nervous-demos.vercel.app) and the building blocks to create your own.

Features GPU acceleration using WebGPU _<sup>WebGPU is currently available behind a flag on [Chrome Canary](https://www.google.com/chrome/canary/) and [Firefox Nightly (untested)](https://www.mozilla.org/en-US/firefox/channel/desktop/)</sup>_.

The JS backend is developed secondarily to the WebGPU backend and is more likely to have issues.

<h2 style="display:inline; margin:0 1rem 1rem 0;">Usage <img width="45" alt="NV sign" style="vertical-align:middle" src="https://user-images.githubusercontent.com/26017543/209094491-6dc7f5aa-4a29-4b89-a06c-969455bbceb5.png"></h2>

```typescript
import nv from "@typio/nervous";

const main = async () => {
  await nv.init({ backend: "webgpu" });

  const [n, m] = await Promise.all([
    nv.random([2047, 512]),
    nv.random([512, 2047]),
  ]);
  let res = await n.matmul(m);

  await res.print(4);
};

main();
```
