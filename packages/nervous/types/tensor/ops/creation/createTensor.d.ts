import { Tensor } from '../../tensor';
export declare enum CreateMethod {
    fill = 0
}
type CommonArgs = {
    method: CreateMethod;
    shape: number[];
};
type FillArgs = CommonArgs & {
    value: number;
};
type DiagArgs = CommonArgs & {
    values: number[];
};
type RandomArgs = CommonArgs & {
    seed: number;
    min: number;
    max: number;
    integer?: boolean;
};
type RandomNormalArgs = CommonArgs & {
    mean: number;
    std: number;
};
type CreateTensorArgs = FillArgs | DiagArgs | RandomArgs | RandomNormalArgs;
export declare const createTensor: (args: CreateTensorArgs) => Tensor;
export {};
