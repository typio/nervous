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

    for (var image = 110; image <= 120; image++) {
        var pixels = [];

        for (var y = 0; y <= 27; y++) {
            for (var x = 0; x <= 27; x++) {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
            }
        }

        var imageData = {};
        imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

        pixelValues.push(imageData);
    }

    let gp;
    p.setup = () => {
        p.createCanvas(512, 512);
        gp = p.createGraphics(p.width, p.height);
        gp.pixelDensity(1);
        p.noLoop();
    };

    p.draw = function () {
        p.clear(0, 0, 0, 0);
        for (let image = 0; image < 20; image++) {
            let randomIndex = Math.floor(Math.random() * 100);

            gp.loadPixels();
            for (var x = 0; x < 28; x++) {
                for (var y = 0; y < 28; y++) {
                    var index = (p.width * y + x) * 4;
                    gp.pixels[index + 0] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 1] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 2] = Object.values(pixelValues[image])[0][28 * y + x];
                    gp.pixels[index + 3] = 255;
                }
            }
            gp.updatePixels();
            p.image(gp, 0 + image * 28,0, p.width, p.width);
            p.text(Object.keys(pixelValues[image])[0], 9 + image * 28 , 50+ (28*image)%10);
        }
    };
};

new p5(s); 