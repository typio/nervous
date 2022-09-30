import p5 from "p5"

import trainImagesFile from './t10k-images.idx3-ubyte?raw';
import labelLabelsFile from './t10k-labels.idx1-ubyte?raw';

const s = p => {
    let pixelValues = [];

    let dataFileBuffer = new ArrayBuffer(trainImagesFile.length);
    let labelFileBuffer = new ArrayBuffer(labelLabelsFile.length);

    for (let i = 0; i < trainImagesFile.length; i++) {
        dataFileBuffer[i] = trainImagesFile.charCodeAt(i);
        labelFileBuffer[i] = labelLabelsFile.charCodeAt(i);
    }

    for (let image = 0; image <= 200; image++) {
        let pixels = [];

        for (let y = 0; y <= 27; y++) {
            for (let x = 0; x <= 27; x++) {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
            }
        }

        let imageData = {};
        imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

        pixelValues.push(imageData);
    }

    let gp;
    p.setup = () => {
        p.createCanvas(560, 1000);
        gp = p.createGraphics(p.width, p.height);
        gp.pixelDensity(1);
        p.noLoop();
    };

    p.draw = function () {
        p.clear(0, 0, 0, 0);
        for (let image = 0; image < 200; image++) {

            gp.loadPixels();
            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    let index = (p.width * y + x) * 4;
                    gp.pixels[index + 0] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 1] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 2] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 3] = 255;
                }
            }
            gp.updatePixels();
            p.image(gp, Math.floor(image%20) * 28, Math.floor(image / 20)*50, p.width, p.height);
            p.text(Object.keys(pixelValues[image])[0], 9 + Math.floor(image%20) * 28 , Math.floor(image / 20)*50+40);
        }
    };
};

new p5(s); 