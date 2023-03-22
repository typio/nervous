import '..';
import type { Tensor } from "../tensor";
export type fnnParams = {
    layers: number[];
    activation: string;
    LR: number;
    stepSize: number;
    batchSize: number;
    epochs: number;
    stopAtAccuracy?: number;
    stopAtLoss?: number;
    stopWhenAccuracyDrops?: number;
    stopWhenLossRises?: number;
    logEvery?: number;
};
export declare const fnn: (train: Tensor[], test: Tensor[], params: fnnParams) => Promise<void>;
