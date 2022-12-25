import { Tensor } from "../tensor";
import type { Rank1To4Array } from "../tensor";
declare const _default: {
    scalar: (value: number) => Tensor;
    tensor: (values: number | Rank1To4Array, shape?: number[]) => Tensor;
    randomNormal: (shape: number[], mean?: number, std?: number) => void;
    add: (a: Tensor, b: Tensor) => Promise<Tensor>;
};
export default _default;
