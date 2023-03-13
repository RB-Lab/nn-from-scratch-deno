export interface Driver {
  // TODO: add fill strategy, e.g. zeros, random, gaussian
  vector(size: number): Vector;
  vector(array: number[]): Vector;
  // TODO: add fill strategy, e.g. zeros, random, gaussian
  matrix(height: number, width: number): Matrix;
  matrix(array: number[][]): Matrix;
  matmul(a: Matrix, b: Matrix): Matrix;
  transpose(a: Matrix): Matrix;
  addVector(a: Matrix, b: Vector): Matrix;
  sumColumns(a: Matrix): Vector;
  copy(a: Matrix): Matrix; // drop this? 🤔
  oneHot(y: Vector, classes: number): Matrix;
  activation: {
    relu: ReluActivation;
    softmax: SoftmaxActivation;
  };
  loss: {
    crossEntropy: CrossEntropyLoss;
  };
  combo: {
    softMaxCrossEntropy: SoftmaxCrossEntropyCombo;
  };
  math: {
    mean: (values: Vector) => number;
  };
}

export type Matrix = number[][];
export type Vector = number[];

export interface ReluActivation {
  forward: (x: Matrix) => Matrix;
  backward: (x: Matrix, grad: Matrix) => Matrix;
}

export interface SoftmaxActivation {
  forward: (x: Matrix) => Matrix;
  backward: (softmaxOutput: Matrix, grad: Matrix) => Matrix;
}

export interface CrossEntropyLoss {
  forward: (yTrue: Vector, yPredBatch: Matrix) => Vector;
  backward: (yTrue: Vector, yPredBatch: Matrix) => Matrix;
}

export interface SoftmaxCrossEntropyCombo {
  forward: (
    yTrue: Vector,
    yPredBatch: Matrix
  ) => { losses: Vector; yPred: Matrix };
  backward: (yTrue: Vector, yPredBatch: Matrix) => Matrix;
}
