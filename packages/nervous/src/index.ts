import { Rank1To4Array, Tensor } from './tensor';

let config = {
    backend: 'js',
    // anything else...?
};

async function createBackend(backend: string) {
    if (backend === 'js') {
        // browser seems to still loads both backends. Can that be fixed with better build process?
        return await import('./backend-js');
    } else if (backend === 'webgpu') {
        return await import('./backend-wepgpu');
    }
}

let backend: any;

export let gpuDevice: null | GPUDevice = null

async function init(userConfig: { backend: string; }) {
    // merge user config with default config
    config = { ...config, ...userConfig };

    backend = await createBackend(config.backend);

    if (config.backend === 'webgpu') {
        try {
            if (!('gpu' in navigator)) {
                console.error("User agent doesn't support WebGPU.");
            }
            const gpuAdapter = await navigator.gpu.requestAdapter();
            if (!gpuAdapter) {
                console.error('No WebGPU adapters found.');
                return null;
            }
            gpuDevice = await gpuAdapter.requestDevice();
            console.log('Initialized GPU device:', gpuDevice);
        } catch (error) {
            console.error(error);
           console.warn('falling back to js backend');
            
            backend = await createBackend('js');
        }
    }
}

// ---------------
// TENSOR CREATION
// ---------------

// Constructors

/**
 * Pass a value
 * ```ts
 * scalar(4)
 * ```
 */
const scalar = (value: number) => backend.default.scalar = (value);

/**
 * Pass a nested array
 * ```ts
 * tensor([[1,2],[3,4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor([1, 2, 3, 4], [2, 2])
 * ```
 */
const tensor = (values: number | Rank1To4Array, shape?: number[]) => backend.default.tensor = (values, shape);


// Generators

/**
 * Pass shape of matrix
 * ```ts
 * fill([2, 2], 1)
 * ```
 */
const fill = (shape: number[], value: number) => backend.default.fill = (shape, value);


/**
 * Pass array of row number and column number, and the position for the one
 * ```ts
 * oneHot([2, 2], [0,1])
 * ```
 * Or flat index
 * ```ts
 * oneHot([2, 2], 1)
 * ```
 */
export const oneHot = (dim: number[] | number, index: number | number[]) => {
    throw new Error('Not implemented.')
}

/**
 * Pass array of values to create 2d diagonal matrix
 */
export const diag = (values: number[]) => {
    // TODO: think about adding custom dimensions or single number values input
    let vLen = values.length
    let m = new Array(vLen * vLen).fill(0)
    let mI = 0
    let vI = 0
    while (vI < vLen) {
        m[mI] = values[vI]
        mI += vLen + 1
        vI++
    }
    return new Tensor(m, [vLen, vLen])
}

/**
 * Pass array of row number and column number
 * ```ts
 * eye([2, 2])
 * ```
 * Or a number for both
 * ```ts
 * eye(2); eye([2])
 * ```
 */
export const eye = (dim: number[] | number, offset?: number) => {
    let rowN: number, colN: number
    if (typeof dim === 'number') {
        dim = [dim]
    }
    rowN = dim[0]
    if (dim.length === 1)
        colN = dim[0]
    else
        colN = dim[1]

    let idx = offset ?? 0
    let values = new Array(rowN * colN).fill(0)
    while (idx < rowN * colN) {
        values[idx] = 1
        idx += colN + 1
    }

    return new Tensor(values, [rowN, colN])
}

/**
 * Pass shape of matrix
 * ```ts
 * random([2, 2])
 * ```
 * And optionally min (inclusive), max (exclusive), and integer
 * ```ts
 * random([2, 2], 0, 10, true)
 * ```
 */
const random = (shape: number[], min?: number, max?: number, integer?: boolean) => backend.default.random(values, shape);

const randomNormal = (shape: number[], mean?: number, std?: number) => backend.default.randomNormal(shape, mean, std);

// -----------------
// TENSOR OPERATIONS
// -----------------

// Elementwise

const add = (a: Tensor, b: Tensor) => backend.default.add(a, b);

// Matrix Product

// const matmul = (a: Tensor, b: Tensor) => backend.default.matmul(a, b);


export default {
    init,
    Tensor,

    tensor,
    scalar,
    random,
    randomNormal,

    add,
};



