import { gpuDevice } from "nervous";
import nv from "nervous"

import matmulWGSL from "./matmul.wgsl?raw";
import addWGSL from "./add.wgsl?raw";


const main = async () => {
    if (gpuDevice === null) throw new Error("gpuDevice is null");

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
                        (4 + await forwardT.shape()[0] * await t2.shape()[1]);

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
                        Math.ceil(await forwardT.shape().at(-1) / 8),
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
                        (4 + await forwardT.shape()[0] * await forwardT.shape()[1]);

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

            const start1 = performance.now();

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
                        await t1.shape()[0] *
                        await t2.shape()[1];

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
                        Math.ceil(await t1.shape().at(-1) / 8),
                        Math.ceil(await t2.shape().at(0) / 8)
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
                        await t1.shape()[0] *
                        await t1.shape()[1];

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
            const end1 = performance.now();
            console.log("this here", end1 - start1);

            const start2 = performance.now();

            // WHY IS WEBGPU COPY SO SLOW IT'S INSANE!!!!!! maybe i messed benchmark up, if not thank GOD i changed impl
            const readGPUBuffer = gpuDevice.createBuffer({
                size:
                    Float32Array.BYTES_PER_ELEMENT *
                    await t1.shape()[0] *
                    await t1.shape()[1],
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
                await t1.shape()[0] *
                await t1.shape()[1]
            );
            gpuDevice.queue.submit([commandEncoder.finish()]);

            await readGPUBuffer.mapAsync(GPUMapMode.READ);

            let result = new Float32Array(
                readGPUBuffer.getMappedRange()
            );
            let forwardT = nv.tensor(result);

            const end2 = performance.now();
            console.log("this thin", end2 - start2);

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

    {
        // Test speed of having all OPs done within one WebGPU code
        // should be bad because OPs can't be independantly parallelized
        console.log(
            `%cTest speed of having all OPs done within one WebGPU 'code'`,
            "background: #aaff44;"
        );
    }
}

main()