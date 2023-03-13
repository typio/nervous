/// <reference types="@webgpu/types" />
//

import { fnn } from './fnn'

const webgpuAvailable = (): boolean => {
    try {
        if ('gpu' in navigator) {
            return true
        }
    } catch {
        return false
    }
    return false
}

export let backend: any

export let gpuAdapter: null | GPUAdapter = null
export let gpuDevice: null | GPUDevice = null

const init = async () => {
    try {
        if (!('gpu' in navigator)) {
            console.error("User agent doesn't support WebGPU.")
        }
        gpuAdapter = await navigator.gpu.requestAdapter()
        if (!gpuAdapter) {
            console.error('No WebGPU adapters found.')
            return null
        }
        gpuDevice = await gpuAdapter.requestDevice()
    } catch (error) {
        console.warn(`${error}, failed to initialize WebGPU`)
    }
}


export default {
    init,
    webgpuAvailable,

    fnn,
}
