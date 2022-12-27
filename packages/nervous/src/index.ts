import { Rank1To4Array, Tensor } from './tensor';

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
    backend: webgpuAvailable() ? 'webgpu' : 'js', // TODO: fix auto-select, it doesnt select webgpu when available
    // anything else...?
};

const createBackend = async (backend: string) => {
    if (backend === 'js') {
        // browser seems to still loads both backends. Can that be fixed with better build process?
        return await import('./backend-js/_index');
    } else if (backend === 'webgpu') {
        return await import('./backend-webgpu/_index');
    } else {
        throw new Error(`Invalid backend, received ${backend}`)
    }
}


let backend: any;

export let gpuDevice: null | GPUDevice = null


const init = async (userConfig?: { backend: string; }) => {
    // merge user config with default config
    config = { ...config, ...userConfig };

    backend = await createBackend(config.backend);
    backend.default = Tensor;

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

            backend = createBackend('js');
            backend.default = Tensor;
        }
    }
}

declare module "./index" {
    interface Tensor {
        matmul(): void;
    }
}

Tensor.prototype.matmul = () => { }

backend.default = Tensor;


// =============================================================================
// TENSOR's
// =============================================================================

// -----------------------------------------------------------------------------
// TENSOR CREATION
// -----------------------------------------------------------------------------

// ------------
// Constructors
// ------------

/** Create scalar tensor from provided number */
const scalar = (value: number): Tensor => backend.default.scalar(value);

/** Construct tensor, pass value array, nested or un-nested, and optional shape */
const tensor = (values: number | Rank1To4Array, shape?: number[]): Tensor => backend.default.tensor(values, shape);

// ----------
// Generators
// ----------

/**Create tensor of provided shape filled with all provided number */
const fill = (shape: number[], value: number): Tensor => backend.default.fill(shape, value);

/**Create tensor of provided shape filled with all zeros */
const zeros = (shape: number | number[]): Tensor => backend.default.zeros(shape)

/**Create tensor of provided shape filled with all ones */
const ones = (shape: number | number[]): Tensor => backend.default.ones(shape)

/** Create one hot tensor of provided shape with 1 at provided index */
export const oneHot = (dim: number[] | number, index: number | number[]): Tensor => backend.default.oneHot(dim, index)

/** Create tensor with non-zero values on diagonals from a provided value array */
export const diag = (values: number[]): Tensor => backend.default.diag(values)

/** Create identity matrix tensor, optional horizontal offset on values  */
export const eye = (dim: number[] | number, offset?: number): Tensor => backend.default.eye(dim, offset)

/** Create tensor of provided shape, filled with random values */
const random = (shape: number[], min?: number, max?: number, integer?: boolean): Tensor => backend.default.random(shape, min, max, integer);

/** Create tensor of provided shape, filled with random values from a normal distribution */
const randomNormal = (shape: number[], mean?: number, std?: number): Tensor => backend.default.randomNormal(shape, mean, std);

// -----------------------------------------------------------------------------
// TENSOR OPERATIONS
// -----------------------------------------------------------------------------

const getValues = (a: Tensor, decimals?: number) => backend.default.getValues(a, decimals);
const getFlatValues = (a: Tensor, decimals?: number) => backend.default.getFlatValues(a, decimals);
const reshape = (a: Tensor, shape: number[]) => backend.default.reshape(a, shape)
const transpose = (a: Tensor) => backend.default.transpose(a)

// --------------
// Matrix Product
// --------------

const matmul = (a: Tensor, b: Tensor): Tensor => backend.default.matmul(a, b);

// -----------
// Elementwise
// -----------

const add = (a: Tensor, b: number | Tensor, axis?: number): Tensor => backend.default.add(a, b, axis);
const minus = (a: Tensor, b: number | Tensor, axis?: number): Tensor => backend.default.minus(a, b, axis);
const mul = (a: Tensor, b: number | Tensor, axis?: number): Tensor => backend.default.mul(a, b, axis);
const div = (a: Tensor, b: number | Tensor, axis?: number): Tensor => backend.default.div(a, b, axis);
const mod = (a: Tensor, b: number | Tensor, axis?: number): Tensor => backend.default.mod(a, b, axis);

// ---------
// Broadcast
// ---------

/** create tensor with relu done to all values  */
const pow = (a: Tensor, exp: number): Tensor => backend.default.pow(a, exp);

/** create tensor with sigmoid done to all values  */
const sigmoid = (a: Tensor): Tensor => backend.default.sigmoid(a);

/** create tensor with softplus done to all values  */
const softplus = (a: Tensor): Tensor => backend.default.softplus(a);

// round(decimals: number) {
// }

// return softmax
const softmax = (a: Tensor): Tensor => backend.default.softmax(a);

/** create tensor with relu done to all values  */
const reLU = (a: Tensor): Tensor => backend.default.reLU(a);

/** create tensor with relu done to all values  */
const gradientReLU = (a: Tensor) => backend.default.gradientReLU(a);


/** create tensor of exponentials of all values on e, or given base  */
const exp = (a: Tensor, base?: number): Tensor => backend.default.exp(a, exp);

/** create tensor of log on all values */
const log = (a: Tensor, base?: number): Tensor => backend.default.log(a);

/** returns tensor with elementwise max of old value vs input number */
const applyMax = (a: Tensor, n: number): Tensor => backend.default.applyMax(a, n)

/** returns tensor with elementwise min of old value vs input number */
const applyMin = (a: Tensor, n: number): Tensor => backend.default.applyMin(a, n)

// ------------
// Calculations
// ------------

const mean = (a: Tensor): number => backend.default.mean(a)

/** return the lp norm as number, default p is 2  */
const lpNorm = (a: Tensor, p?: number): number => backend.default.lpNorm(a, p)

/** return Frobenius Norm as number, represents the size of a matrix */
const fNorm = (a: Tensor): number => backend.default.fNorm(a)

/** returns sum of diagonal elements as number */
const trace = (a: Tensor): number => backend.default.trace(a)

/** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
const sum = (a: Tensor, axis?: 0 | 1): Tensor => backend.default.sum(a, axis)

const getmax = (a: Tensor, axis?: 0 | 1): Tensor => backend.default.getmax(a, axis)

/** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
const getmin = (a: Tensor, axis?: 0 | 1): Tensor => backend.default.getmin(a, axis)

const argmax = (a: Tensor): Tensor => backend.default.getmin(a, argmax)

const argmin = (a: Tensor): Tensor => backend.default.getmin(a, argmin)

export default {
    init,
    Tensor,

    add,
    applyMax,
    applyMin,
    argmax,
    argmin,
    diag,
    div,
    exp,
    eye,
    fNorm,
    fill,
    getFlatValues,
    getValues,
    getmax,
    getmin,
    gradientReLU,
    log,
    lpNorm,
    matmul,
    mean,
    minus,
    mod,
    mul,
    oneHot,
    ones,
    pow,
    random,
    randomNormal,
    reLU,
    reshape,
    scalar,
    sigmoid,
    softmax,
    softplus,
    sum,
    tensor,
    trace,
    transpose,
    zeros,
};



