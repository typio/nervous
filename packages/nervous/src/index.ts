/// <reference types="@webgpu/types" />
//

export let gpuAdapter: null | GPUAdapter = null
export let gpuDevice: null | GPUDevice = null

export const webgpuAvailable = (): boolean => {
    try {
        if ('gpu' in navigator) {
            return true
        }
    } catch {
        return false
    }
    return false
}

export const init = async () => {
    try {
        if (!('gpu' in navigator)) {
            console.error("User agent doesn't support WebGPU.")
            return
        }
        gpuAdapter = await navigator.gpu.requestAdapter()
        if (!gpuAdapter) {
            console.error('No WebGPU adapters found.')
            return
        }
        gpuDevice = await gpuAdapter.requestDevice()
    } catch (error) {
        console.warn(`${error}, failed to initialize WebGPU`)
    }
}


export * from './tensor/_index'

 export * from './fnn'
