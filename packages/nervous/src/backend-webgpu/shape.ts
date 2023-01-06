import { Tensor } from "../tensor"
import { toArr } from "../tensorUtils";

export const shape = (a: Tensor): number[] => {

    // remove trailing 1's in shape segement of data
    let i = 3;
    while (i > 0 && (a.data[i] === 0)) {
        i--;
    }
    return toArr(a.data.slice(0, i + 1));
}