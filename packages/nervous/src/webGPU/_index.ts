export const createMappedBuffer = (device: GPUDevice, data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer => {
    const gpuBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, data.byteLength),
        usage: usage,
    });
    (data instanceof Float32Array
        ? new Float32Array(gpuBuffer.getMappedRange())
        : new Uint32Array(gpuBuffer.getMappedRange())
    ).set(data);
    gpuBuffer.unmap();

    return gpuBuffer;
}

export const runComputeShader = (gpuDevice: GPUDevice, buffers: GPUBuffer[], code: string, workgroup_size: number[]): GPUBuffer => {
    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({ code }),
            entryPoint: 'main',
        },
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, index) => ({
            binding: index,
            resource: { buffer },
        }))
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(workgroup_size[0], workgroup_size[1], workgroup_size[2])
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])

    return buffers[buffers.length - 1]
}
