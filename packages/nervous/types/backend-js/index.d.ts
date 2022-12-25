import { Rank1To4Array, Tensor } from "../tensor";
declare const _default: {
    scalar: (value: number) => Tensor;
    tensor: (values: number | Rank1To4Array, shape?: number[]) => Tensor;
    fill: (shape: number | number[], value: number) => Tensor;
    zeros: (shape: number | number[]) => Tensor;
    ones: (shape: number | number[]) => Tensor;
    random: (shape: number[], min?: number, max?: number, integer?: boolean) => Tensor;
    randomNormal: (shape: number[], mean?: number, std?: number) => Tensor;
    add: (a: Tensor, b: Tensor) => Tensor;
};
export default _default;
