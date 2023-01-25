import { Rank1To4Array, Tensor } from './tensor'

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

let config = {
    backend: 'auto',
    // anything else...?
}

const createBackend = async (backend: string) => {
    if (backend === 'js') {
        // browser seems to still loads both backends. Can that be fixed with better build process?
        return await import('./backend-js/_index')
    } else if (backend === 'webgpu') {
        return await import('./backend-webgpu/_index')
    } else {
        throw new Error(`Invalid backend, received ${backend}`)
    }
}

export let backend: any

export let gpuDevice: null | GPUDevice = null

const init = async (userConfig?: { backend: string }) => {
    // merge user config with default config
    let local_config = { ...config, ...userConfig }

    if (local_config.backend === 'auto') {
        local_config.backend = 'webgpu'
    }

    backend = await createBackend(local_config.backend)

    if (local_config.backend === 'webgpu') {
        try {
            if (!('gpu' in navigator)) {
                console.error("User agent doesn't support WebGPU.")
            }
            const gpuAdapter = await navigator.gpu.requestAdapter()
            if (!gpuAdapter) {
                console.error('No WebGPU adapters found.')
                return null
            }
            gpuDevice = await gpuAdapter.requestDevice()
        } catch (error) {
            console.warn(`${error}, falling back to js backend`)

            backend = await createBackend('js')
        }
    }
}

// =============================================================================
// TENSOR CREATION
// =============================================================================

// ------------
// Constructors
// ------------

/** Create scalar tensor from provided number */
const scalar = (value: number): Tensor => backend.default.scalar(value)

/** Construct tensor, pass value array, nested or un-nested, and optional shape */
const tensor = (values: number | Rank1To4Array, shape?: number[]): Tensor => backend.default.tensor(values, shape)

// ----------
// Generators
// ----------

/**Create tensor of provided shape filled with all provided number */
const fill = async (shape: number[], value: number): Promise<Tensor> => backend.default.fill(shape, value)

/**Create tensor of provided shape filled with all zeros */
const zeros = async (shape: number | number[]): Promise<Tensor> => backend.default.zeros(shape)

/**Create tensor of provided shape filled with all ones */
const ones = async (shape: number | number[]): Promise<Tensor> => backend.default.ones(shape)

/** Create one hot tensor of provided shape with 1 at provided index */
const oneHot = async (dim: number[] | number, index: number | number[]): Promise<Tensor> =>
    backend.default.oneHot(dim, index)

/** Create tensor with non-zero values on diagonals from a provided value array */
const diag = async (values: number[]): Promise<Tensor> => backend.default.diag(values)

/** Create identity matrix tensor, optional horizontal offset on values  */
const eye = async (dim: number[] | number, offset?: number): Promise<Tensor> => backend.default.eye(dim, offset)

/** Create tensor of provided shape, filled with random values */
const random = async (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean): Promise<Tensor> =>
    backend.default.random(shape, seed, min, max, integer)

/** Create tensor of provided shape, filled with random values from a normal distribution */
const randomNormal = async (shape: number[], seed?: number, mean?: number, std?: number): Promise<Tensor> =>
    backend.default.randomNormal(shape, seed, mean, std)

export default {
    init,
    webgpuAvailable,
    Tensor,

    scalar,
    tensor,
    eye,
    diag,
    ones,
    zeros,
    fill,
    oneHot,
    random,
    randomNormal,
}
