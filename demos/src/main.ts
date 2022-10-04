import pako from 'pako'
import p5 from "p5"

import * as lt from "light"


// import trainImagesFile from './t10k-images-idx3-ubyte.gz.data'
// import labelLabelsFile from './t10k-labels-idx1-ubyte.gz.data'

let trainImagesFilePromise = import('./t10k-images-idx3-ubyte.gz.data')
let trainlabelLabelsPromise = import('./t10k-labels-idx1-ubyte.gz.data')


// trainImagesFilePromise.then((result) => {
//     console.log(result.default)
// })


const s = (p) => {



    let pixelValues = []

    trainImagesFilePromise.then((data) => {
        let dataFileBuffer = pako.ungzip(data.default)
        trainlabelLabelsPromise.then((data) => {
            let labelFileBuffer = pako.ungzip(data.default)

            for (let image = 0; image <= 1000; image++) {
                let pixels = []

                for (let y = 0; y < 28; y++) {
                    for (let x = 0; x < 28; x++) {
                        pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16])
                    }
                }

                let imageData = {}
                imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels

                pixelValues.push(imageData)

            }

        })

    })







    let gp
    p.setup = () => {
        p.createCanvas(1120, 1000)
        gp = p.createGraphics(p.width, p.height)
        gp.pixelDensity(1)
        p.noLoop()

        window.document.getElementById('loading_msg').style = 'display: none;'


    }

    p.draw = function () {
        p.clear(0, 0, 0, 0)


        for (let image = 0; image <= 1000; image++) {
            gp.reset()
            gp.loadPixels()
            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    let index = (p.width * y + x) * 4
                    gp.pixels[index + 0] = Object.values(pixelValues[image])[0][(28 * y) + x]
                    gp.pixels[index + 1] = Object.values(pixelValues[image])[0][(28 * y) + x]
                    gp.pixels[index + 2] = Object.values(pixelValues[image])[0][(28 * y) + x]
                    gp.pixels[index + 3] = 255
                }
            }
            gp.updatePixels()
            p.image(gp, Math.floor(image % 40) * 28, Math.floor(image / 40) * 50, p.width, p.height)
            p.text(Object.keys(pixelValues[image])[0], 9 + Math.floor(image % 40) * 28, Math.floor(image / 40) * 50 + 40)
        }



    }
}

new p5(s) 