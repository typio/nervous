import { Rank1To4Array, Tensor } from "../../tensor";
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
    min?: number;
    max?: number;
    integer: boolean;
    mean?: number;
    std?: number;
};
type CreateTensorArgs = FillArgs | DiagArgs | RandomArgs;
declare enum CreateMethod {
    fill = 0,
    diag = 1,
    random = 2
}
export declare const scalar: (value: number) => Tensor;
export declare const tensor: (value: number | Rank1To4Array, shape?: number[]) => Tensor;
export declare const fill: (shape: number[], value: number) => Tensor;
export declare const zeros: (shape: number[]) => Tensor;
export declare const ones: (shape: number[]) => Tensor;
export declare const diag: (values: number[]) => Tensor;
export declare const eye: (shape: number[]) => Tensor;
export declare const random: (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean) => Tensor;
export declare const randomNormal: (shape: number[], seed?: number, mean?: number, std?: number, integer?: boolean) => Tensor;
export declare const createTensor: (args: CreateTensorArgs) => Tensor;
export {};
