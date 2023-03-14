import { Driver, Matrix, Vector } from "./driver/driver.ts";

type Activation =
  | Driver["activation"]["relu"]
  | Driver["activation"]["softmax"];
export interface BaseOptions {
  driver: Driver;
  activation?: Activation;
}
export interface LoadOption extends BaseOptions {
  weights: Matrix;
  biases: Vector;
}
export interface NewOptions extends BaseOptions {
  inputSize: number;
  neuronsNumber: number;
}

export class DenseLayer {
  #driver: Driver;
  #weights: Matrix;
  #biases: Vector;
  #activation?: Activation;
  #input?: Matrix;
  #gradW?: Matrix;
  #gradB?: Vector;
  #gradInput?: Matrix;
  constructor(options: LoadOption | NewOptions) {
    this.#driver = options.driver;
    this.#activation = options.activation;
    if ("weights" in options) {
      if (options.weights[0].length !== options.biases.length) {
        throw new Error(`Weights width must be equal to biases length`);
      }
      this.#weights = options.weights;
      this.#biases = options.biases;
    } else {
      this.#weights = this.#driver.matrix(
        options.inputSize,
        options.neuronsNumber
      );
      this.#biases = this.#driver.vector(options.neuronsNumber);
    }
  }

  forward(input: Matrix): Matrix {
    this.#input = input;
    const result = this.#driver.addVector(
      this.#driver.matmul(input, this.#weights),
      this.#biases
    );
    return this.#activation ? this.#activation.forward(result) : result;
  }

  backward(output: Matrix, grad: Matrix): Matrix {
    if (!this.#input) {
      throw new Error(`Layer must be run forward before running backward`);
    }
    if (this.#activation) {
      grad = this.#activation.backward(output, grad);
    }
    this.#gradW = this.#driver.matmul(
      this.#driver.transpose(this.#input),
      grad
    );
    this.#gradB = this.#driver.sumColumns(grad);
    this.#gradInput = this.#driver.matmul(
      grad,
      this.#driver.transpose(this.#weights)
    );
    return this.#gradInput;
  }

  update(learningRate: number): void {
    if (!this.#gradW || !this.#gradB) {
      throw new Error(`Layer must be run backward before running update`);
    }
    this.#weights = this.#weights.map((row, i) =>
      row.map((value, j) => value - learningRate * this.#gradW![i][j])
    );
    this.#biases = this.#biases.map(
      (value, i) => value - learningRate * this.#gradB![i]
    );
  }
}
