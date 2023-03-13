import { DenseLayer } from "./lib/dense_layer.ts";
import { simpleDriver } from "./lib/driver/simple_driver.ts";
import spiral from "../nn-from-scratch-py/spiral.json" assert { type: "json" };

const { X, y } = spiral;

const inputLayer = new DenseLayer({
  driver: simpleDriver,
  inputSize: 2,
  neuronsNumber: 64,
  activation: simpleDriver.activation.relu,
});

const outputLayer = new DenseLayer({
  driver: simpleDriver,
  inputSize: 64,
  neuronsNumber: 3,
});

const out: number[][] = [];
const learningRate = 1;
for (let epoch = 0; epoch < 10000; epoch++) {
  const layer1output = inputLayer.forward(X);
  const output = outputLayer.forward(layer1output);

  const {losses, yPred} = simpleDriver.combo.softMaxCrossEntropy.forward(y, output);
  // const losses = simpleDriver.losses.crossEntropy.forward(y, output);

  const meanLoss = simpleDriver.math.mean(losses);

  const correctPredictions = yPred
    .map((row, i) => argMax(row) === y[i])
    .filter((y) => y);
  const accuracy = correctPredictions.length / y.length;

  out.push([epoch, accuracy, meanLoss]);
  const gradLoss = simpleDriver.combo.softMaxCrossEntropy.backward(y, yPred);
  // const gradLoss = simpleDriver.losses.crossEntropy.backward(y, output);
  const gradOut = outputLayer.backward(output, gradLoss);
  inputLayer.backward(layer1output, gradOut);
  outputLayer.update(learningRate);
  inputLayer.update(learningRate);
}

console.log(JSON.stringify(out));

function argMax(array: number[]) {
  return array.indexOf(Math.max(...array));
}
