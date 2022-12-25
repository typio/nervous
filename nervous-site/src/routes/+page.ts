import nv from "nervous";

import { browser } from '$app/environment';

if (browser) {

    const main = async () => {
        let start = performance.now();
        await nv.init({ backend: "webgpu" });

        // let tensor1 = await nv.randomNormal([2, 2]);
        // let tensor2 = new nv.Tensor([
        //     [9, 9],
        //     [5, 6],
        // ]);
        let tensor2 = nv.tensor([
            [9, 9],
            [5, 6],
        ]);

        let added = await nv.add(tensor2, tensor2);
        console.log('add result:', added);
    };

    main();
}