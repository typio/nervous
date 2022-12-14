// import pako from 'pako'
// import p5 from "p5"

// import * as nv from "nervous"

// let trainImagesFilePromise = import('./train-images-idx3-ubyte.gz.data')
// let trainlabelLabelsPromise = import('./train-labels-idx1-ubyte.gz.data')
// let testImagesFilePromise = import('./t10k-images-idx3-ubyte.gz.data')
// let testlabelLabelsPromise = import('./t10k-labels-idx1-ubyte.gz.data')

// let trainPixels: number[][] = new Array(60000)
// let trainLabels: number[] = new Array(60000)
// trainImagesFilePromise.then((data) => {
//     let dataFileBuffer = pako.ungzip(data.default)
//     trainlabelLabelsPromise.then((data) => {
//         let labelFileBuffer = pako.ungzip(data.default)

//         for (let image = 0; image < 60000; image++) {
//             let pixels = new Array(784)
//             for (let y = 0; y < 28; y++) {
//                 for (let x = 0; x < 28; x++) {
//                     pixels[x + (y * 28)] = (dataFileBuffer[(image * 784) + (x + (y * 28)) + 16])
//                 }
//             }
//             trainPixels[image] = pixels
//             trainLabels[image] = labelFileBuffer[image + 8]
//         }
//     })
// })

// let testPixels: number[][] = new Array(10000)
// let testLabels: number[] = new Array(10000)
// testImagesFilePromise.then((data) => {
//     let dataFileBuffer = pako.ungzip(data.default)
//     testlabelLabelsPromise.then((data) => {
//         let labelFileBuffer = pako.ungzip(data.default)

//         for (let image = 0; image < 10000; image++) {
//             let pixels = new Array(784)
//             for (let y = 0; y < 28; y++) {
//                 for (let x = 0; x < 28; x++) {
//                     pixels[x + (y * 28)] = dataFileBuffer[(image * 784) + (x + (y * 28)) + 16]
//                 }
//             }
//             testPixels[image] = pixels
//             testLabels[image] = labelFileBuffer[image + 8]
//         }
//     })
// })

// let layer_dims = [784, 16, 10]
// class Layer {
//     weights: nv.Tensor
//     biases: nv.Tensor
//     output: nv.Tensor

//     constructor(inputsLength, neuronsLength) {
//         this.weights = nv.random([inputsLength, neuronsLength])
//         console.log('weights', this.weights)

//         this.biases = nv.zeroes([1, neuronsLength])
//     }

//     forward(inputs: nv.Tensor): nv.Tensor {
//         return (inputs.dot(this.weights)).add(this.biases)
//     }
// }

// class ReLU {
//     inputs: nv.Tensor

//     constructor(inputs: nv.Tensor) {
//         this.inputs = inputs
//     }

//     forward = () => {
//         return this.inputs.applyMax(0)
//     }
// }

// class Softmax {
//     inputs: nv.Tensor

//     constructor(inputs: nv.Tensor) {
//         this.inputs = inputs
//     }

//     forward = () => {
//         console.log('this.inputs', this.inputs)
//         console.log('this.inputs.getMax(1)', this.inputs.getMax(1))
//         console.log('this.inputs.minus(this.inputs.getMax(1))', this.inputs.minus(this.inputs.getMax(1)))

//         let exp_values = (this.inputs.minus(this.inputs.getMax(1))).exp()
//         console.log('exp_values', exp_values)

//         console.log('exp_values.sum(1)', exp_values.sum(1))
//         return exp_values.div(exp_values.sum(1))
//     }
// }


// const s = (p) => {
//     let gp
//     p.setup = () => {
//         p.createCanvas(1120, 1000)
//         gp = p.createGraphics(p.width, p.height)
//         gp.pixelDensity(1)
//         p.noLoop()

//         let thing = window.document.getElementById('loading_msg')
//         if (thing !== null)
//             thing.innerHTML = "Loaded."

//         let batch_inputs = nv.tensor(trainPixels.slice(0, 64))
//         let batch_labels = nv.tensor(trainLabels.slice(0, 64))

//         let layer1_weights = nv.random([784, 16])
//         let layer1_biases = nv.random([16, 1])
//         console.log(layer1_biases)


//         let layer2_weights = nv.random([16, 10])
//         let layer2_biases = nv.random([1, 10])



//         console.log(batch_inputs.dot(layer1_weights).add(layer1_biases))

//         // let firstLayer = new Layer(layer_dims[0], layer_dims[1])
//         // let secondLayer = new Layer(layer_dims[1], layer_dims[2])


//         // console.log(
//         //     new Softmax(secondLayer.forward(new ReLU(firstLayer.forward(batch)).forward())).forward()
//         // )


//     }

//     p.draw = function () {
//         p.clear(0, 0, 0, 0)

//         // for (let image = 0; image <= 1000; image++) {
//         //     gp.reset()
//         //     gp.loadPixels()
//         //     for (let x = 0; x < 28; x++) {
//         //         for (let y = 0; y < 28; y++) {
//         //             let index = (p.width * y + x) * 4
//         //             let shade = trainPixels[image][(28 * y) + x]
//         //             gp.pixels[index + 0] = shade
//         //             gp.pixels[index + 1] = shade
//         //             gp.pixels[index + 2] = shade
//         //             gp.pixels[index + 3] = 255
//         //         }
//         //     }
//         //     gp.updatePixels()
//         //     p.image(gp, Math.floor(image % 40) * 28, Math.floor(image / 40) * 50, p.width, p.height)
//         //     p.text(trainLabels[image], 9 + Math.floor(image % 40) * 28, Math.floor(image / 40) * 50 + 40)
//         // }
//     }
// }

// new p5(s) 