import { Tensor } from "../tensor";
import { Activation } from '../layer';
export type fnnHyperParams = {
    layers: number[];
    activation: Activation;
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
export declare const fnn: (train: Tensor[], test: Tensor[], params: fnnHyperParams) => Promise<void>;
