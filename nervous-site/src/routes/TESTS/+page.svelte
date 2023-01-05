<script lang="ts">
    import * as PIXI from "pixi.js";
    import nv from "nervous";
    import { browser } from "$app/environment";

    import tests from "./tests";

    import { gpuDevice } from "nervous";

    import matmulWGSL from "./matmul.wgsl?raw";
    import addWGSL from "./add.wgsl?raw";

    let backend = "auto";

    let testResults: any[] = [];

    const runTests = async () => {
        await nv.init({ backend });

        testResults = [];
        for (let i = 0; i < tests.length; i++) {
            let suite = tests[i];
            testResults.push({
                name: suite.suite,
                results: [],
            });

            for (let j = 0; j < suite.tests.length; j++) {
                // testResults.suite.push(test.code());
                let test = suite.tests[j];
                testResults[i].results.push({
                    name: test.name,
                    res: null,
                });
                testResults = testResults;

                let codeErrored = false;
                const codeRes = await (async () => {
                    try {
                        return JSON.stringify(await test.code());
                    } catch (e) {
                        codeErrored = true;
                        return e;
                    }
                })();

                let expectsErrored = false;
                const expectsRes = await (async () => {
                    try {
                        return JSON.stringify(await test.expects());
                    } catch (e) {
                        expectsErrored = true;
                        return e;
                    }
                })();

                testResults[i].results[j] = {
                    ...testResults[i].results[j],
                    res: codeRes === expectsRes,
                    code: test.code,
                    expects: test.expects,
                    codeErrored,
                    expectsErrored,
                    codeRes,
                    expectsRes,
                };
            }
        }
    };

    if (browser) {
        // runTests();
        const main = async () => {

            {
                await nv.init()
                const size = 512;
                let result = await nv.random([size, size]);
                let resultValues = result.values();

                const app = new PIXI.Application({
                    width: 512,
                    height: 512,
                    antialias: true,
                });

                document.body.appendChild(app.view);
                const graphics = new PIXI.Graphics();
                for (let i = 0; i < result.shape()[0]; i++) {
                    for (let j = 0; j < result.shape()[1]; j++) {
                        if (i % 64 === 0 || j % 64 === 0)
                            graphics.beginFill(0x000000);
                        else graphics.beginFill(0xff0000 * resultValues[i][j]);

                        graphics.drawRect(i, j, 1, 1);
                        graphics.endFill();
                    }
                }
                app.stage.addChild(graphics);
            }
            
            // Benchmarking methods of chaining WebGPU OPs together such as would be done in a neural network
            console.log(
                `%cBenchmarking methods of chaining WebGPU OPs together such as would be done in a neural network`,
                "background: #7DF9FF;"
            );
            let mSize = 255;
            let steps = 100;
            let iter = 1;

            await nv.init({ backend: "js" });

            let t1 = await nv.random([mSize, mSize]);
            let t2 = (await nv.random([mSize, mSize])).mul(0.001);
            let t3 = await nv.random([mSize, mSize]);

            await nv.init({ backend: "webgpu" });
            let tensor = await (
                await nv.tensor([
                    [0.6183, 0.1661, 0.2896, 0.8502, 0.5295],
                    [0.2598, 0.3651, 0.0412, 0.3813, 0.3422],
                    [0.6018, 0.4518, 0.7268, 0.8983, 0.8653],
                    [0.5311, 0.2222, 0.5785, 0.5307, 0.865],
                    [0.6808, 0.1407, 0.8364, 0.1303, 0.4623],
                ])
            ).toGPU();

            
            let tensor2 = await (
                await nv.tensor([
                    [0.8974, 0.6172],
                    [0.4255, 0.068],
                    [0.662, 0.2339],
                    [0.1782, 0.8271],
                    [0.7705, 0.9918],
                ])
            ).toGPU();


            console.log(JSON.stringify(await (await (await tensor.matmul(tensor2)).toJS()).values()));

            {
                // Test speed of OP reading value to CPU and passing value in JS to next OP
                console.log(
                    `%cTest speed of OP reading value to CPU and passing value in JS to next OP`,
                    "background: #aaff44;"
                );

                let timesElapsed: number[] = [];

                for (let i = 0; i < iter; i++) {
                    const startTime = performance.now();
                    let forwardT = t1;

                    for (let j = 0; j < steps; j++) {
                        // forwardT *= t2
                        {
                            const forwardTGPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: forwardT.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(
                                forwardTGPUBuffer.getMappedRange()
                            ).set(forwardT.data);
                            forwardTGPUBuffer.unmap();

                            const t2GPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: t2.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(t2GPUBuffer.getMappedRange()).set(
                                t2.data
                            );
                            t2GPUBuffer.unmap();

                            let resultSize =
                                Float32Array.BYTES_PER_ELEMENT *
                                (4 + forwardT.shape()[0] * t2.shape()[1]);

                            const resultGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.STORAGE |
                                    GPUBufferUsage.COPY_SRC,
                            });

                            const readGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.COPY_DST |
                                    GPUBufferUsage.MAP_READ,
                            });

                            const computePipeline =
                                gpuDevice.createComputePipeline({
                                    layout: "auto",
                                    compute: {
                                        module: gpuDevice.createShaderModule({
                                            code: matmulWGSL,
                                        }),
                                        entryPoint: "main",
                                    },
                                });

                            const bindGroup = gpuDevice.createBindGroup({
                                layout: computePipeline.getBindGroupLayout(0),
                                entries: [
                                    {
                                        binding: 0,
                                        resource: {
                                            buffer: forwardTGPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 1,
                                        resource: {
                                            buffer: t2GPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 2,
                                        resource: {
                                            buffer: resultGPUBuffer,
                                        },
                                    },
                                ],
                            });

                            const commandEncoder =
                                gpuDevice.createCommandEncoder();
                            const passEncoder =
                                commandEncoder.beginComputePass();
                            passEncoder.setPipeline(computePipeline);
                            passEncoder.setBindGroup(0, bindGroup);
                            passEncoder.dispatchWorkgroups(
                                Math.ceil(forwardT.shape().at(-1) / 8),
                                Math.ceil(t2.shape().at(0) / 8)
                            );
                            passEncoder.end();

                            commandEncoder.copyBufferToBuffer(
                                resultGPUBuffer,
                                0,
                                readGPUBuffer,
                                0,
                                resultSize
                            );
                            gpuDevice.queue.submit([commandEncoder.finish()]);
                            await readGPUBuffer.mapAsync(GPUMapMode.READ);

                            let result = new Float32Array(
                                readGPUBuffer.getMappedRange()
                            );
                            forwardT = nv.tensor(result);
                        }

                        // forwardT += t3
                        {
                            const forwardTGPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: forwardT.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(
                                forwardTGPUBuffer.getMappedRange()
                            ).set(forwardT.data);
                            forwardTGPUBuffer.unmap();

                            const t3GPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: t3.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(t3GPUBuffer.getMappedRange()).set(
                                t3.data
                            );
                            t3GPUBuffer.unmap();

                            let resultSize =
                                Float32Array.BYTES_PER_ELEMENT *
                                (4 + forwardT.shape()[0] * forwardT.shape()[1]);

                            const resultGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.STORAGE |
                                    GPUBufferUsage.COPY_SRC,
                            });

                            const readGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.COPY_DST |
                                    GPUBufferUsage.MAP_READ,
                            });

                            const computePipeline =
                                gpuDevice.createComputePipeline({
                                    layout: "auto",
                                    compute: {
                                        module: gpuDevice.createShaderModule({
                                            code: addWGSL,
                                        }),
                                        entryPoint: "main",
                                    },
                                });

                            const bindGroup = gpuDevice.createBindGroup({
                                layout: computePipeline.getBindGroupLayout(0),
                                entries: [
                                    {
                                        binding: 0,
                                        resource: {
                                            buffer: forwardTGPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 1,
                                        resource: {
                                            buffer: t3GPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 2,
                                        resource: {
                                            buffer: resultGPUBuffer,
                                        },
                                    },
                                ],
                            });

                            const commandEncoder =
                                gpuDevice.createCommandEncoder();
                            const passEncoder =
                                commandEncoder.beginComputePass();
                            passEncoder.setPipeline(computePipeline);
                            passEncoder.setBindGroup(0, bindGroup);
                            passEncoder.dispatchWorkgroups(
                                forwardT.flatValues().length
                            );
                            passEncoder.end();

                            commandEncoder.copyBufferToBuffer(
                                resultGPUBuffer,
                                0,
                                readGPUBuffer,
                                0,
                                resultSize
                            );
                            gpuDevice.queue.submit([commandEncoder.finish()]);
                            await readGPUBuffer.mapAsync(GPUMapMode.READ);

                            let result = new Float32Array(
                                readGPUBuffer.getMappedRange()
                            );
                            forwardT = nv.tensor(result);
                        }
                    }
                    console.log(forwardT.data);

                    const endTime = performance.now();
                    timesElapsed.push(endTime - startTime);
                }

                let avgTime =
                    timesElapsed.reduce((a, b) => a + b) / timesElapsed.length;

                console.log(
                    `%ctime taken: ${Math.round(avgTime)}ms`,
                    "background: #ffee88;"
                );
            }

            {
                // Test speed of OP returning buffer and passing it to next OP
                console.log(
                    `%cTest speed of OP returning buffer and passing it to next OP`,
                    "background: #aaff44;"
                );

                let timesElapsed: number[] = [];

                for (let i = 0; i < iter; i++) {
                    const startTime = performance.now();
                    const forwardTGPUBuffer = gpuDevice.createBuffer({
                        mappedAtCreation: true,
                        size: t1.data.byteLength,
                        usage:
                            GPUBufferUsage.STORAGE |
                            GPUBufferUsage.COPY_DST |
                            GPUBufferUsage.COPY_SRC,
                    });
                    new Float32Array(forwardTGPUBuffer.getMappedRange()).set(
                        t1.data
                    );
                    forwardTGPUBuffer.unmap();

                    for (let j = 0; j < steps; j++) {
                        // forwardT *= t2
                        {
                            const t2GPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: t2.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(t2GPUBuffer.getMappedRange()).set(
                                t2.data
                            );
                            t2GPUBuffer.unmap();

                            let resultSize =
                                Float32Array.BYTES_PER_ELEMENT *
                                t1.shape()[0] *
                                t2.shape()[1];

                            const resultGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.STORAGE |
                                    GPUBufferUsage.COPY_SRC,
                            });

                            // const readGPUBuffer = gpuDevice.createBuffer({
                            //     size: resultSize,
                            //     usage:
                            //         GPUBufferUsage.COPY_DST |
                            //         GPUBufferUsage.MAP_READ,
                            // });

                            const computePipeline =
                                gpuDevice.createComputePipeline({
                                    layout: "auto",
                                    compute: {
                                        module: gpuDevice.createShaderModule({
                                            code: matmulWGSL,
                                        }),
                                        entryPoint: "main",
                                    },
                                });

                            const bindGroup = gpuDevice.createBindGroup({
                                layout: computePipeline.getBindGroupLayout(0),
                                entries: [
                                    {
                                        binding: 0,
                                        resource: {
                                            buffer: forwardTGPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 1,
                                        resource: {
                                            buffer: t2GPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 2,
                                        resource: {
                                            buffer: resultGPUBuffer,
                                        },
                                    },
                                ],
                            });

                            const commandEncoder =
                                gpuDevice.createCommandEncoder();
                            const passEncoder =
                                commandEncoder.beginComputePass();
                            passEncoder.setPipeline(computePipeline);
                            passEncoder.setBindGroup(0, bindGroup);
                            passEncoder.dispatchWorkgroups(
                                Math.ceil(t1.shape().at(-1) / 8),
                                Math.ceil(t2.shape().at(0) / 8)
                            );
                            passEncoder.end();

                            commandEncoder.copyBufferToBuffer(
                                resultGPUBuffer,
                                0,
                                forwardTGPUBuffer,
                                0,
                                resultSize
                            );
                            gpuDevice.queue.submit([commandEncoder.finish()]);
                            // await readGPUBuffer.mapAsync(GPUMapMode.READ);

                            // let result = new Float32Array(
                            //     readGPUBuffer.getMappedRange()
                            // );
                            // forwardT = nv.tensor(result);
                        }

                        // forwardT += t3
                        {
                            const t3GPUBuffer = gpuDevice.createBuffer({
                                mappedAtCreation: true,
                                size: t3.data.byteLength,
                                usage: GPUBufferUsage.STORAGE,
                            });
                            new Float32Array(t3GPUBuffer.getMappedRange()).set(
                                t3.data
                            );
                            t3GPUBuffer.unmap();

                            let resultSize =
                                Float32Array.BYTES_PER_ELEMENT *
                                t1.shape()[0] *
                                t1.shape()[1];

                            const resultGPUBuffer = gpuDevice.createBuffer({
                                size: resultSize,
                                usage:
                                    GPUBufferUsage.STORAGE |
                                    GPUBufferUsage.COPY_SRC,
                            });

                            // const readGPUBuffer = gpuDevice.createBuffer({
                            //     size: resultSize,
                            //     usage:
                            //         GPUBufferUsage.COPY_DST |
                            //         GPUBufferUsage.MAP_READ,
                            // });

                            const computePipeline =
                                gpuDevice.createComputePipeline({
                                    layout: "auto",
                                    compute: {
                                        module: gpuDevice.createShaderModule({
                                            code: addWGSL,
                                        }),
                                        entryPoint: "main",
                                    },
                                });

                            const bindGroup = gpuDevice.createBindGroup({
                                layout: computePipeline.getBindGroupLayout(0),
                                entries: [
                                    {
                                        binding: 0,
                                        resource: {
                                            buffer: forwardTGPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 1,
                                        resource: {
                                            buffer: t3GPUBuffer,
                                        },
                                    },
                                    {
                                        binding: 2,
                                        resource: {
                                            buffer: resultGPUBuffer,
                                        },
                                    },
                                ],
                            });

                            const commandEncoder =
                                gpuDevice.createCommandEncoder();
                            const passEncoder =
                                commandEncoder.beginComputePass();
                            passEncoder.setPipeline(computePipeline);
                            passEncoder.setBindGroup(0, bindGroup);
                            passEncoder.dispatchWorkgroups(
                                t1.flatValues().length
                            );
                            passEncoder.end();

                            commandEncoder.copyBufferToBuffer(
                                resultGPUBuffer,
                                0,
                                forwardTGPUBuffer,
                                0,
                                resultSize
                            );
                            gpuDevice.queue.submit([commandEncoder.finish()]);
                        }
                    }

                    const readGPUBuffer = gpuDevice.createBuffer({
                        size:
                            Float32Array.BYTES_PER_ELEMENT *
                            t1.shape()[0] *
                            t1.shape()[1],
                        usage:
                            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    });
                    const commandEncoder = gpuDevice.createCommandEncoder();
                    commandEncoder.copyBufferToBuffer(
                        forwardTGPUBuffer,
                        0,
                        readGPUBuffer,
                        0,
                        Float32Array.BYTES_PER_ELEMENT *
                            t1.shape()[0] *
                            t1.shape()[1]
                    );
                    gpuDevice.queue.submit([commandEncoder.finish()]);

                    await readGPUBuffer.mapAsync(GPUMapMode.READ);

                    let result = new Float32Array(
                        readGPUBuffer.getMappedRange()
                    );
                    let forwardT = nv.tensor(result);
                    console.log(forwardT.data);

                    const endTime = performance.now();
                    timesElapsed.push(endTime - startTime);
                }

                let avgTime =
                    timesElapsed.reduce((a, b) => a + b) / timesElapsed.length;

                console.log(
                    `%ctime taken: ${Math.round(avgTime)}ms`,
                    "background: #ffee88;"
                );
            }

            {
                // Test speed of having all OPs done within one WebGPU bind layout
                console.log(
                    `%cTest speed of having all OPs done within one WebGPU bind layout`,
                    "background: #aaff44;"
                );
            }

            // should be bad because OPs can't be independantly parallelized
            // {
            //     // Test speed of having all OPs done within one WebGPU code
            //     console.log(
            //         `%cTest speed of having all OPs done within one WebGPU code`,
            //         "background: #aaff44;"
            //     );
            // }
            

            {
                await nv.init({ backend: "webgpu" });
                const size = 512;
                console.log(
                    `%cspeed test of random([${size}, ${size}])`,
                    "background: #00ff00;"
                );
                let start = performance.now();

                let tensor1 = await nv.random([size, size], 42);
                let tensor2 = await nv.random([size, size], 69);
                let mid = performance.now();

                {
                    let result = await tensor1.matmul(tensor2);
                    console.log(result);
                }
                console.log(
                    `%cwebgpu time: ${mid - start}ms`,
                    "background: #ffff00;"
                );

                await nv.init({ backend: "js" });
                mid = performance.now();

                tensor1 = await nv.random([size, size]);
                tensor2 = await nv.random([size, size]);
                let end = performance.now();

                {
                    let result = await tensor1.matmul(tensor2);
                    console.log(result);
                }

                console.log(
                    `%cjs time: ${end - mid}ms`,
                    "background: #ffff00;"
                );
            }
        };
        main();
    }
</script>

<head>
    <title>Tests</title>
</head>

<nav class="mt-6 ml-6">
    <a class="text-3xl text-red-600" href="/">Demos</a>
</nav>

<label for="backend-select">Backend: </label>
<select name="backend" id="backend-select" bind:value={backend}>
    <option value="auto">Auto</option>
    <option value="js">JS</option>
    <option value="webgpu" disabled={!nv.webgpuAvailable()}>WebGPU</option>
</select>

<button
    class="rounded shadow text-slate-100 bg-green-500 active:bg-green-600 p-2"
    on:click={runTests}
>
    Run Tests
</button>

{#if testResults.length !== 0}
    <button
        class="rounded shadow text-slate-100 bg-red-500 active:bg-red-600 p-2"
        on:click={() => {
            testResults = [];
        }}
    >
        Clear Results
    </button>
{/if}

{#each testResults as suite}
    {#if suite}
        <h3 class="text-xl font-medium">{suite.name}</h3>
        {#each suite.results || [] as result}
            <div class="inline-flex">
                <p>{result.name}</p>
                <pre
                    class="px-2 max-h-28 overflow-scroll py-1 ml-4 text-sm bg-slate-900 text-slate-300">{result.code}</pre>
            </div>
            <div>
                {#if result.res === null}
                    <p>running</p>
                {:else}
                    <!-- <p>{result.res}</p> -->

                    {#if result.codeErrored || result.expectsErrored}
                        <div class="bg-red-300">
                            {#if result.codeErrored}
                                ❌ Test {result.codeRes}
                            {/if}
                        </div>
                        <div class="bg-red-300">
                            {#if result.expectsErrored}
                                ❌ Expected Result Calculation {result.expectsRes}
                            {/if}
                        </div>
                    {:else if result.res}
                        <div class="bg-green-300">
                            ✅ {result.codeRes} ≡ {result.expectsRes}
                        </div>
                    {:else}
                        <div class="bg-red-300">
                            ❌ {result.codeRes} ≢ {result.expectsRes}
                        </div>
                    {/if}
                {/if}
            </div>
        {/each}
    {/if}
{/each}

<p>is this random enough?</p>
